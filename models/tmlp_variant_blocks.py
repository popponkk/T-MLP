import math
import time
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias]
            )
            x = x + bias[None]
        return x


class TMLPBackbone(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        n_layers: int = 1,
        d_token: int = 1024,
        d_ffn_factor: float = 0.66,
        ffn_dropout: float | None = None,
        residual_dropout: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)
        n_tokens = self.tokenizer.n_tokens

        def make_normalization(d=d_token):
            return nn.LayerNorm(d)

        d_hidden = int(d_token * d_ffn_factor)
        self.d_token = d_token
        self.d_ffn = d_hidden

        class SGU(nn.Module):
            def __init__(self, n_token):
                super().__init__()
                self.proj = nn.Linear(n_token, n_token)
                self.norm = make_normalization(d=d_hidden)

            def forward(self, x):
                u, v = torch.chunk(x, 2, -1)
                v = self.norm(v).transpose(1, 2)
                v = self.proj(v).transpose(1, 2)
                return u * v

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'linear0': nn.Linear(d_token, d_hidden * 2),
                    'sgu': SGU(n_tokens),
                    'linear1': nn.Linear(d_hidden, d_token),
                }
            )
            self.layers.append(layer)

        self.activation = F.gelu
        self.normalization = make_normalization()
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def encode_tokens(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = x
            x = self.normalization(x)
            x = layer['linear0'](x)
            x = self.activation(x)
            x = layer['sgu'](x)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            x = layer['linear1'](x)
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            x = x + x_residual
        return x

    def encode_hidden(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> tuple[Tensor, Tensor]:
        token_states = self.encode_tokens(x_num, x_cat)
        hidden = token_states[:, 0]
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        return hidden, token_states


class SRHead(nn.Module):
    def __init__(self, d_token: int, d_out: int):
        super().__init__()
        mid_dim = max(d_token // 4, 16)
        self.base_head = nn.Linear(d_token, d_out)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, d_out),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, hidden: Tensor):
        y_base = self.base_head(hidden)
        delta_y = self.residual_head(hidden)
        alpha = torch.sigmoid(self.alpha_logit)
        prediction = y_base + alpha * delta_y
        return prediction, {
            'y_base': y_base,
            'delta_y': delta_y,
            'alpha': alpha,
        }


class PEEHead(nn.Module):
    def __init__(self, d_token: int, d_out: int, n_heads: int = 4, learned_weight: bool = False):
        super().__init__()
        mid_dim = max(d_token // 4, 16)
        self.base_head = nn.Linear(d_token, d_out)
        self.learned_weight = learned_weight
        self.parallel_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, d_out),
            )
            for _ in range(n_heads)
        ])
        self.alpha_logit = nn.Parameter(torch.tensor(-4.0))
        if learned_weight:
            self.confidence_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_token),
                    nn.Linear(d_token, mid_dim),
                    nn.GELU(),
                    nn.Linear(mid_dim, 1),
                )
                for _ in range(n_heads)
            ])

    def forward(self, hidden: Tensor):
        y_base = self.base_head(hidden)
        deltas = torch.stack([head(hidden) for head in self.parallel_heads], dim=1)
        if self.learned_weight:
            logits = torch.stack([head(hidden) for head in self.confidence_heads], dim=1)
            weights = torch.softmax(logits, dim=1)
            delta_y = (weights * deltas).sum(dim=1)
        else:
            weights = None
            delta_y = deltas.mean(dim=1)
        alpha = torch.sigmoid(self.alpha_logit)
        prediction = y_base + alpha * delta_y
        return prediction, {
            'y_base': y_base,
            'delta_y': delta_y,
            'all_deltas': deltas,
            'head_weights': weights,
            'alpha': alpha,
        }


class LatentGroupRefiner(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.n_groups = n_groups
        self.group_assign = nn.Parameter(torch.randn(n_features, n_groups) * 0.01)
        self.group_gate = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, max(d_token // 4, 16)),
            nn.GELU(),
            nn.Linear(max(d_token // 4, 16), 1),
        )
        self.fusion = nn.Parameter(torch.tensor(0.1))
        self.group_dropout = nn.Dropout(dropout)

    def forward(self, token_states: Tensor) -> tuple[Tensor, dict]:
        cls_token = token_states[:, :1]
        feature_tokens = token_states[:, 1:]
        assign = torch.softmax(self.group_assign, dim=-1)
        group_tokens = torch.einsum('bfd,fg->bgd', feature_tokens, assign)
        gates = torch.sigmoid(self.group_gate(group_tokens))
        gated_groups = self.group_dropout(group_tokens * gates)
        feature_residual = torch.einsum('bgd,fg->bfd', gated_groups, assign)
        refined_features = feature_tokens + self.fusion * feature_residual
        refined_tokens = torch.cat([cls_token, refined_features], dim=1)
        return refined_tokens, {
            'group_assign': assign,
            'group_gates': gates,
            'group_tokens': gated_groups,
        }


class RegressionVariantTrainer(TabModel):
    def __init__(self):
        super().__init__()

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        return model_config

    def compute_variant_loss(self, predictions, targets, extras, training_args):
        loss = F.huber_loss(
            predictions.squeeze(-1),
            targets,
            delta=training_args.get('huber_delta', 1.0),
        )
        alpha_reg = training_args.get('alpha_reg', 1e-4)
        delta_reg = training_args.get('delta_reg', 1e-4)
        group_reg = training_args.get('group_reg', 1e-5)
        if 'alpha' in extras:
            loss = loss + alpha_reg * extras['alpha'].pow(2)
        if 'delta_y' in extras:
            loss = loss + delta_reg * extras['delta_y'].pow(2).mean()
        if 'group_assign' in extras:
            entropy = -(extras['group_assign'] * extras['group_assign'].clamp_min(1e-8).log()).sum(dim=-1).mean()
            loss = loss + group_reg * entropy
        return loss

    def fit(
        self,
        train_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None,
        X_num: ty.Optional[torch.Tensor] = None,
        X_cat: ty.Optional[torch.Tensor] = None,
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None,
        eval_set: ty.Tuple[torch.Tensor, np.ndarray] = None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        if task != 'regression':
            raise NotImplementedError(f'{self.base_name} currently supports regression only')
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault('save_path', f'results/{self.base_name}')
        meta_args.setdefault('log_every_n_epochs', 50)
        check_dir(meta_args['save_path'])
        self.meta_config = meta_args

        if training_args is None:
            training_args = {}
        training_args.setdefault('optimizer', 'adamw')
        training_args.setdefault('batch_size', 64)
        training_args.setdefault('max_epochs', 10000)
        training_args.setdefault('patience', patience)
        training_args.setdefault('save_frequency', 'epoch')
        training_args.setdefault('huber_delta', 1.0)
        training_args.setdefault('alpha_reg', 1e-4)
        training_args.setdefault('delta_reg', 1e-4)
        training_args.setdefault('group_reg', 1e-5)
        self.training_config = training_args

        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)
        if train_loader is not None:
            train_loader, placeholders = train_loader
            training_args['batch_size'] = train_loader.batch_size
        else:
            train_loader, placeholders = TabModel.prepare_tensor_loader(
                X_num=X_num, X_cat=X_cat, ys=ys, ids=ids,
                batch_size=training_args['batch_size'],
                shuffle=True,
            )

        if eval_set is not None:
            dev_loader = TabModel.prepare_tensor_loader(
                X_num=eval_set[0][0], X_cat=eval_set[0][1], ys=eval_set[0][2], ids=eval_set[0][3],
                batch_size=training_args['batch_size'],
            )
            test_loader = None
            if len(eval_set) == 2:
                test_loader = TabModel.prepare_tensor_loader(
                    X_num=eval_set[1][0], X_cat=eval_set[1][1], ys=eval_set[1][2], ids=eval_set[1][3],
                    batch_size=training_args['batch_size'],
                )
        else:
            dev_loader, test_loader = None, None

        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0.0
        for _ in range(training_args['max_epochs']):
            self.model.train()
            tot_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                predictions, extras = self.model(x_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time
                loss = self.compute_variant_loss(predictions, y, extras, training_args)
                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
            if training_args['save_frequency'] == 'epoch':
                is_early_stop = self.save_evaluate_dnn(
                    tot_step, steps_per_epoch, tot_loss, tot_time,
                    task, training_args['patience'], meta_args['save_path'],
                    dev_loader, y_std, test_loader=test_loader,
                )
                if is_early_stop:
                    self.save(meta_args['save_path'])
                    self.load_best_dnn(meta_args['save_path'])
                    return
        self.save(meta_args['save_path'])
        self.load_best_dnn(meta_args['save_path'])

    def predict(
        self,
        dev_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None,
        X_num: ty.Optional[torch.Tensor] = None,
        X_cat: ty.Optional[torch.Tensor] = None,
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None,
        task: str = None,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: ty.Optional[dict] = None,
    ):
        def inference_step(model, x_num, x_cat):
            predictions, _ = model(x_num, x_cat, return_extras=False)
            return predictions, 0.0

        return self.dnn_predict(
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            y_std=y_std,
            ids=ids,
            task=task,
            return_probs=return_probs,
            return_metric=return_metric,
            return_loss=return_loss,
            meta_args=meta_args,
        )

    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)
