import copy
import math
import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_token = int(d_token)
        self.weight = nn.Parameter(torch.empty(self.n_features, self.d_token))
        self.bias = nn.Parameter(torch.empty(self.n_features, self.d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        ffn_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, d_token),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(h)

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + self.ffn_dropout(h)
        return x


class TJEPAEncoder(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        n_layers: int,
        ffn_hidden_dim: int,
        dropout: float,
        n_reg_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.n_reg_tokens = int(n_reg_tokens)
        self.reg_tokens = (
            nn.Parameter(torch.empty(self.n_reg_tokens, d_token))
            if self.n_reg_tokens > 0
            else None
        )
        if self.reg_tokens is not None:
            nn.init.normal_(self.reg_tokens, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if self.reg_tokens is not None:
            reg = self.reg_tokens.unsqueeze(0).expand(tokens.size(0), -1, -1)
            tokens = torch.cat([tokens, reg], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class TJEPAPredictor(nn.Module):
    def __init__(self, d_token: int, predictor_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden_dim, d_token),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TJEPARegressor(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        d_out: int = 1,
        d_token: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_hidden_dim: int = 512,
        predictor_hidden_dim: int = 512,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
        n_reg_tokens: int = 4,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.d_out = int(d_out)
        self.mask_ratio = float(mask_ratio)
        self.ema_decay = float(ema_decay)
        self.n_reg_tokens = int(n_reg_tokens)

        self.tokenizer = NumericFeatureTokenizer(self.n_num_features, d_token)
        self.mask_token = nn.Parameter(torch.zeros(d_token))
        self.context_encoder = TJEPAEncoder(
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            n_reg_tokens=n_reg_tokens,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = TJEPAPredictor(
            d_token=d_token,
            predictor_hidden_dim=predictor_hidden_dim,
            dropout=dropout,
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_out),
        )
        self._aux_loss = torch.tensor(0.0)

    def get_aux_loss(self) -> Tensor:
        if isinstance(self._aux_loss, Tensor):
            return self._aux_loss
        return torch.tensor(float(self._aux_loss))

    @torch.no_grad()
    def update_target_encoder(self):
        for context_param, target_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(self.ema_decay).add_(
                context_param.data, alpha=1.0 - self.ema_decay
            )

    def _resolve_x_num(
        self,
        x_num: ty.Union[Tensor, dict, list, tuple, None],
        x_cat: ty.Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(x_num, dict):
            if "x_num" in x_num:
                x_num = x_num["x_num"]
            elif "x" in x_num:
                x_num = x_num["x"]
        elif isinstance(x_num, (list, tuple)):
            if len(x_num) == 0:
                x_num = None
            else:
                x_num = x_num[0]
        if x_num is None:
            raise ValueError("TJEPARegressor requires numerical features as x_num.")
        return x_num.float()

    def _sample_target_mask(self, batch_size: int, n_features: int, device: torch.device) -> Tensor:
        n_target = int(round(n_features * self.mask_ratio))
        n_target = max(1, min(n_features - 1, n_target))
        noise = torch.rand(batch_size, n_features, device=device)
        target_indices = noise.topk(k=n_target, dim=1).indices
        target_mask = torch.zeros(batch_size, n_features, device=device, dtype=torch.bool)
        target_mask.scatter_(1, target_indices, True)
        return target_mask

    def _gather_targets(self, tokens: Tensor, target_indices: Tensor) -> Tensor:
        gather_index = target_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        return tokens.gather(dim=1, index=gather_index)

    def forward(
        self,
        x_num: ty.Union[Tensor, dict, list, tuple, None],
        x_cat: ty.Optional[Tensor] = None,
    ) -> Tensor:
        x_num = self._resolve_x_num(x_num, x_cat)
        tokens = self.tokenizer(x_num)
        batch_size, n_features, _ = tokens.shape

        if not self.training:
            self._aux_loss = tokens.new_zeros(())
            context_repr = self.context_encoder(tokens)
            pooled = context_repr[:, :n_features, :].mean(dim=1)
            y_pred = self.regression_head(pooled)
            return y_pred.squeeze(-1)

        target_mask = self._sample_target_mask(batch_size, n_features, tokens.device)
        target_indices = target_mask.float().topk(k=target_mask.sum(dim=1).max().item(), dim=1).indices

        masked_tokens = tokens.clone()
        masked_tokens[target_mask] = self.mask_token

        context_repr = self.context_encoder(masked_tokens)
        with torch.no_grad():
            self.target_encoder.eval()
            target_repr = self.target_encoder(tokens).detach()

        context_feature_repr = context_repr[:, :n_features, :]
        target_feature_repr = target_repr[:, :n_features, :]

        pred_input = self._gather_targets(context_feature_repr, target_indices)
        pred_target = self.predictor(pred_input)
        target_latent = self._gather_targets(target_feature_repr, target_indices)
        self._aux_loss = F.mse_loss(pred_target, target_latent)

        pooled = context_feature_repr.mean(dim=1)
        y_pred = self.regression_head(pooled)
        return y_pred.squeeze(-1)


class TJEPABaseline(TabModel):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = "cuda",
        feat_gate: ty.Optional[str] = None,
        pruning: ty.Optional[str] = None,
        dataset=None,
    ):
        if feat_gate or pruning:
            raise NotImplementedError("tjepa does not support sparse gating options")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.aux_loss_weight = float(model_config.pop("aux_loss_weight", 1.0))
        self.model = TJEPARegressor(
            n_num_features=n_num_features,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "tjepa"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.setdefault("d_token", 256)
        model_config.setdefault("n_layers", 4)
        model_config.setdefault("n_heads", 8)
        model_config.setdefault("ffn_hidden_dim", 512)
        model_config.setdefault("predictor_hidden_dim", 512)
        model_config.setdefault("dropout", 0.1)
        model_config.setdefault("mask_ratio", 0.3)
        model_config.setdefault("n_reg_tokens", 4)
        model_config.setdefault("ema_decay", 0.996)
        model_config.setdefault("aux_loss_weight", 1.0)
        return model_config

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
        if task != "regression":
            raise NotImplementedError("tjepa currently supports regression only")

        if meta_args is None:
            meta_args = {}
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        training_args = {} if training_args is None else training_args
        training_args.setdefault("optimizer", "adamw")
        training_args.setdefault("weight_decay", 0.0)
        training_args.setdefault("batch_size", 64)
        training_args.setdefault("max_epochs", 10000)
        training_args.setdefault("save_frequency", "epoch")
        training_args.setdefault("patience", patience)
        self.training_config = training_args

        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)

        if train_loader is not None:
            train_loader, placeholders = train_loader
            training_args["batch_size"] = train_loader.batch_size
        else:
            train_loader, placeholders = TabModel.prepare_tensor_loader(
                X_num=X_num,
                X_cat=X_cat,
                ys=ys,
                ids=ids,
                batch_size=training_args["batch_size"],
                shuffle=True,
            )

        if eval_set is not None:
            dev_eval_set = eval_set[0]
            dev_loader = TabModel.prepare_tensor_loader(
                X_num=dev_eval_set[0],
                X_cat=dev_eval_set[1],
                ys=dev_eval_set[2],
                ids=dev_eval_set[3],
                batch_size=training_args["batch_size"],
            )
            test_loader = None
            if len(eval_set) == 2:
                test_eval_set = eval_set[1]
                test_loader = TabModel.prepare_tensor_loader(
                    X_num=test_eval_set[0],
                    X_cat=test_eval_set[1],
                    ys=test_eval_set[2],
                    ids=test_eval_set[3],
                    batch_size=training_args["batch_size"],
                )
        else:
            dev_loader, test_loader = None, None

        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0.0
        for _ in range(training_args["max_epochs"]):
            self.model.train()
            tot_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                logits = self.model(x_num, x_cat)
                forward_time = time.time() - start_time
                supervised_loss = TabModel.compute_loss(logits, y, task)
                aux_loss = self.model.get_aux_loss()
                loss = supervised_loss + self.aux_loss_weight * aux_loss

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                self.gradient_policy()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                self.model.update_target_encoder()

                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                tot_time += forward_time + backward_time

            if training_args["save_frequency"] == "epoch":
                is_early_stop = self.save_evaluate_dnn(
                    tot_step,
                    steps_per_epoch,
                    tot_loss,
                    tot_time,
                    task,
                    training_args["patience"],
                    meta_args["save_path"],
                    dev_loader,
                    y_std,
                    test_loader=test_loader,
                )
                if is_early_stop:
                    self.save(meta_args["save_path"])
                    self.load_best_dnn(meta_args["save_path"])
                    return

        self.save(meta_args["save_path"])
        self.load_best_dnn(meta_args["save_path"])

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
            start_time = time.time()
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time
            return logits, used_time

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
        self.save_config(output_dir)
        self.save_history(output_dir)


Model = TJEPABaseline
