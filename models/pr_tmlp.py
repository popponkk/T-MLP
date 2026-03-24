import math
import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .sparser import Sparser, make_sparser


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


class ResidualExpert(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self._init_small()

    def _init_small(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden)


class _PRTMLP(nn.Module):
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
        d_out: int = 1,
        num_prototypes: int = 4,
        routing_tau: float = 1.0,
        residual_dropout_head: float = 0.1,
        pure_experts: bool = False,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError('PR-TMLP currently supports regression only')

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)
        n_tokens = self.tokenizer.n_tokens
        self.hidden_dim = d_token
        self.num_prototypes = num_prototypes
        self.routing_tau = routing_tau
        self.pure_experts = pure_experts

        def make_normalization(d=d_token):
            return nn.LayerNorm(d)

        d_hidden = int(d_token * d_ffn_factor)

        class SGU(nn.Module):
            def __init__(self, n_token):
                super().__init__()
                self.proj = nn.Linear(n_token, n_token)
                self.norm = make_normalization(d=d_hidden)

            def forward(self, x, z=None):
                u, v = torch.chunk(x, 2, -1)
                v = self.norm(v).transpose(1, 2)
                v = self.proj(v).transpose(1, 2)
                if z is not None:
                    return u * (
                        z * v
                        + (1 - z) * torch.ones_like(v, device=v.device)
                    )
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
        self.base_head = nn.Linear(d_token, d_out)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_token) * 0.02)
        self.residual_heads = nn.ModuleList(
            [ResidualExpert(d_token, residual_dropout_head) for _ in range(num_prototypes)]
        )

    def _sp_linear(self, lin: nn.Linear, x, z=None):
        if z is None:
            x = x @ lin.weight.T
        else:
            x = x @ (torch.diag(z) @ lin.weight.T)
        if lin.bias is not None:
            x = x + lin.bias
        return x

    def _sp_residual(self, x, x_residual, z=None):
        if z is None:
            return x + x_residual
        return z * x + x_residual

    def _encode_hidden(
        self,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        sparser: ty.Optional[Sparser] = None,
    ) -> Tensor:
        if sparser is not None:
            zs = sparser(x_num, x_cat)
        else:
            zs = {}
        if isinstance(x_num, tuple):
            x_num = x_num[1]
        x = self.tokenizer(x_num, x_cat)

        if sparser is not None:
            zs.update(sparser(x, is_raw_input=False))
        if 'feature_z' in zs:
            x = x * zs['feature_z']

        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = x
            x = self.normalization(x)
            hidden_z = zs.get('hidden_z')
            x = self._sp_linear(layer['linear0'], x, hidden_z)
            x = self.activation(x)
            sgu_z = zs['sgu_z'][layer_idx] if 'sgu_z' in zs else None
            x = layer['sgu'](x, sgu_z)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            int_z = zs['intermediate_z'][layer_idx] if 'intermediate_z' in zs else None
            x = self._sp_linear(layer['linear1'], x, int_z)
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            layer_z = zs['layer_z'][layer_idx] if 'layer_z' in zs else None
            x = self._sp_residual(x, x_residual, layer_z)

        hidden = x[:, 0]
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        return hidden

    def _forward_from_hidden(self, hidden: Tensor, return_aux: bool = False):
        diff = hidden[:, None, :] - self.prototypes[None, :, :]
        similarity = -(diff.pow(2).sum(dim=-1))
        routing = torch.softmax(similarity / self.routing_tau, dim=-1)

        residual_stack = torch.cat([head(hidden) for head in self.residual_heads], dim=1)
        routed_residual = (routing * residual_stack).sum(dim=1, keepdim=True)
        y_base = self.base_head(hidden)
        y_hat = routed_residual if self.pure_experts else y_base + routed_residual

        if not return_aux:
            return y_hat.squeeze(-1)

        pairwise_dist = torch.cdist(self.prototypes, self.prototypes, p=2)
        off_diag = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=pairwise_dist.device)
        pairwise_vals = pairwise_dist[off_diag]
        sparse_entropy = -(routing * routing.clamp_min(1e-8).log()).sum(dim=-1).mean()
        avg_usage = routing.mean(dim=0)
        mean_abs_residual = routed_residual.abs().mean()

        return y_hat.squeeze(-1), {
            'routing_weights': routing,
            'routing_entropy': sparse_entropy,
            'prototype_usage': avg_usage,
            'prototype_distances': pairwise_vals,
            'prototype_distance_mean': pairwise_vals.mean() if pairwise_vals.numel() > 0 else torch.zeros((), device=hidden.device),
            'routed_residual': routed_residual,
            'mean_abs_residual': mean_abs_residual,
            'y_base': y_base,
            'residual_stack': residual_stack,
        }

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_aux: bool = False):
        hidden = self._encode_hidden(x_num, x_cat, sparser=None)
        return self._forward_from_hidden(hidden, return_aux=return_aux)

    def _sp_forward(
        self,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        sparser: Sparser,
        return_aux: bool = False,
    ):
        hidden = self._encode_hidden(x_num, x_cat, sparser=sparser)
        return self._forward_from_hidden(hidden, return_aux=return_aux)


class PRTMLP(TabModel):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = 'cuda',
        feat_gate: ty.Optional[str] = None,
        pruning: ty.Optional[str] = None,
        dataset=None,
        variant_name: str = 'pr_tmlp_residual',
    ):
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.variant_name = variant_name
        self.model = _PRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        if any([feat_gate, pruning]):
            self.sparser = make_sparser(self.model, pruning, feat_gate, device, dataset)
        self.base_name = variant_name
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.setdefault('num_prototypes', 4)
        model_config.setdefault('routing_tau', 1.0)
        model_config.setdefault('residual_dropout_head', 0.1)
        model_config.setdefault('pure_experts', False)
        return model_config

    @staticmethod
    def _as_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return x

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
            raise NotImplementedError('pr_tmlp variants currently support regression only')
        if training_args is None:
            training_args = {}
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault('save_path', f'results/{self.base_name}')
        meta_args.setdefault('log_every_n_epochs', 50)
        check_dir(meta_args['save_path'])
        self.meta_config = meta_args

        training_args.setdefault('optimizer', 'adamw')
        training_args.setdefault('batch_size', 64)
        training_args.setdefault('max_epochs', 10000)
        training_args.setdefault('patience', patience)
        training_args.setdefault('save_frequency', 'epoch')
        training_args.setdefault('huber_delta', 1.0)
        training_args.setdefault('lambda_sparse', 1e-3)
        training_args.setdefault('lambda_balance', 1e-2)
        training_args.setdefault('lambda_sep', 1e-4)
        self.training_config = training_args

        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)
        if self.sparser is not None:
            l0_optimizer, lagrangian_optimizer = self.sparser.make_optimizer(1e-3)

        if train_loader is not None:
            train_loader, placeholders = train_loader
            training_args['batch_size'] = train_loader.batch_size
        else:
            train_loader, placeholders = TabModel.prepare_tensor_loader(
                X_num=X_num,
                X_cat=X_cat,
                ys=ys,
                ids=ids,
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
            if self.sparser is not None:
                self.sparser.train()
            tot_loss = 0.0
            entropy_vals = []
            usage_vals = []
            distance_vals = []
            residual_vals = []
            for batch in train_loader:
                optimizer.zero_grad()
                if self.sparser is not None:
                    l0_optimizer.zero_grad()
                    lagrangian_optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)

                start_time = time.time()
                if self.sparser is None:
                    predictions, aux = self.model(x_num, x_cat, return_aux=True)
                else:
                    predictions, aux = self.model._sp_forward(
                        x_num, x_cat, self.sparser, return_aux=True
                    )
                forward_time = time.time() - start_time

                reg_loss = F.huber_loss(
                    predictions,
                    y,
                    delta=training_args['huber_delta'],
                )
                sparse_loss = aux['routing_entropy']
                balance_loss = ((aux['prototype_usage'] - 1.0 / self.model.num_prototypes) ** 2).sum()
                sep_loss = torch.exp(-aux['prototype_distances'].pow(2)).mean() if aux['prototype_distances'].numel() > 0 else torch.zeros((), device=predictions.device)
                loss = reg_loss
                loss = loss + training_args['lambda_sparse'] * sparse_loss
                loss = loss + training_args['lambda_balance'] * balance_loss
                loss = loss + training_args['lambda_sep'] * sep_loss
                if self.sparser is not None:
                    loss = loss + self.sparser.regularization(tot_step)

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if self.sparser is not None:
                    l0_optimizer.step()
                    lagrangian_optimizer.step()
                    if self.sparser.l0_module is not None:
                        self.sparser.l0_module.constrain_parameters()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                entropy_vals.append(float(aux['routing_entropy'].detach().cpu().item()))
                usage_vals.append(aux['prototype_usage'].detach().cpu().numpy())
                distance_vals.append(float(aux['prototype_distance_mean'].detach().cpu().item()))
                residual_vals.append(float(aux['mean_abs_residual'].detach().cpu().item()))

            if training_args['save_frequency'] == 'epoch':
                train_metrics = self.predict(
                    X_num=X_num,
                    X_cat=X_cat,
                    ys=ys,
                    y_std=y_std,
                    task=task,
                    return_metric=True,
                    return_loss=False,
                )[1]['metrics']
                val_metrics = None
                if eval_set is not None:
                    val_metrics = self.predict(
                        X_num=eval_set[0][0],
                        X_cat=eval_set[0][1],
                        ys=eval_set[0][2],
                        y_std=y_std,
                        task=task,
                        return_metric=True,
                        return_loss=False,
                    )[1]['metrics']
                distance_snapshot = (
                    aux['prototype_distances'].detach().cpu().tolist()
                    if aux['prototype_distances'].numel() > 0 else []
                )
                self.append_log(
                    meta_args['save_path'],
                    (
                        f"[metrics] train_rmse={train_metrics['rmse']:.6g}"
                        f" | train_mae={train_metrics['mae']:.6g}"
                        f" | train_r2={train_metrics['r2']:.6g}"
                        + (
                            f" | val_rmse={val_metrics['rmse']:.6g}"
                            f" | val_mae={val_metrics['mae']:.6g}"
                            f" | val_r2={val_metrics['r2']:.6g}"
                            if val_metrics is not None else ""
                        )
                    ),
                )
                self.append_log(
                    meta_args['save_path'],
                    (
                        f"[pr] routing_entropy={np.mean(entropy_vals):.6g}"
                        f" | prototype_usage={np.mean(np.stack(usage_vals, axis=0), axis=0).tolist()}"
                        f" | prototype_distance_mean={np.mean(distance_vals):.6g}"
                        f" | prototype_distances={distance_snapshot}"
                        f" | mean_abs_residual={np.mean(residual_vals):.6g}"
                    ),
                )
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
        def inference_step(model, x_num, x_cat, sparser=None):
            start = time.time()
            if sparser is None:
                predictions = model(x_num, x_cat, return_aux=False)
            else:
                predictions = model._sp_forward(x_num, x_cat, sparser, return_aux=False)
            used = time.time() - start
            return predictions, used

        return self.dnn_predict(
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            ids=ids,
            y_std=y_std,
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
