"""HingeMix-DP: GGPL hinge tokens with directed dual-projection relations.

This file intentionally owns all model-specific components.  It only uses the
project's generic ``TabModel`` training/checkpoint infrastructure.
"""

import json
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


def _activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported slimtok activation: {name}")


class GGPLTokenizer(nn.Module):
    """The GGPL tokenizer copied from the formal implementation unchanged."""

    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
        num_breakpoints: int,
        learnable_breakpoints: bool = True,
    ) -> None:
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.d_token = int(d_token)
        self.num_breakpoints = int(num_breakpoints)
        self.learnable_breakpoints = bool(learnable_breakpoints)

        if categories is None:
            d_bias = self.d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = self.d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape}")

        self.cls_token = nn.Parameter(Tensor(d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        if self.d_numerical > 0:
            basis_dim = self.num_breakpoints + 1
            self.basis_weight = nn.Parameter(
                torch.empty(self.d_numerical, basis_dim, d_token)
            )
            self.basis_bias = nn.Parameter(torch.zeros(self.d_numerical, d_token))
            self.register_buffer("breakpoint_min", torch.zeros(self.d_numerical, 1))
            self.breakpoint_delta_raw = nn.Parameter(
                torch.zeros(self.d_numerical, self.num_breakpoints)
            )
            if not self.learnable_breakpoints:
                self.breakpoint_delta_raw.requires_grad_(False)
        else:
            self.register_buffer("breakpoint_min", torch.zeros(0, 1))
            self.basis_weight = None
            self.basis_bias = None
            self.breakpoint_delta_raw = None
        self._init_weights()

    def _init_weights(self) -> None:
        nn_init.kaiming_uniform_(self.cls_token.unsqueeze(0), a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        if self.basis_weight is not None:
            nn_init.normal_(self.basis_weight, mean=0.0, std=1e-3)
            nn_init.zeros_(self.basis_bias)

    @property
    def n_tokens(self) -> int:
        return 1 + self.d_numerical + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    @staticmethod
    def _inverse_softplus(x: Tensor) -> Tensor:
        return torch.log(torch.expm1(torch.clamp(x, min=1e-6)))

    def set_breakpoints(self, breakpoints: Tensor) -> None:
        if self.d_numerical == 0:
            return
        breakpoints = torch.sort(breakpoints.detach().float(), dim=1).values
        first = breakpoints[:, :1]
        deltas = torch.diff(breakpoints, dim=1, prepend=first)
        deltas[:, 0:1] = 1e-3
        deltas = torch.clamp(deltas, min=1e-4)
        self.breakpoint_min = (first - deltas[:, :1]).to(self.breakpoint_min.device)
        self.breakpoint_delta_raw.data.copy_(
            self._inverse_softplus(deltas).to(self.breakpoint_delta_raw.device)
        )

    def breakpoints(self) -> Tensor:
        if self.d_numerical == 0:
            return torch.zeros(0, self.num_breakpoints, device=self.cls_token.device)
        deltas = F.softplus(self.breakpoint_delta_raw)
        return self.breakpoint_min.to(deltas.device) + torch.cumsum(deltas, dim=1)

    def _numeric_tokens(self, x_num: ty.Optional[Tensor]) -> ty.Optional[Tensor]:
        if self.d_numerical == 0 or x_num is None:
            return None
        breakpoints = self.breakpoints().to(x_num.device)
        basis = F.relu(x_num.unsqueeze(-1) - breakpoints.unsqueeze(0))
        basis = torch.cat([x_num.unsqueeze(-1), basis], dim=-1)
        return torch.einsum("bdf,dft->bdt", basis, self.basis_weight) + self.basis_bias

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_num is not None else x_cat
        assert x_some is not None
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(len(x_some), 1, -1)
        pieces = [cls]
        numeric_tokens = self._numeric_tokens(x_num)
        if numeric_tokens is not None:
            pieces.append(numeric_tokens)
        if x_cat is not None:
            if self.category_embeddings is None or self.category_offsets is None:
                raise ValueError("Categorical input was supplied but this tokenizer has no categories")
            pieces.append(self.category_embeddings(x_cat + self.category_offsets[None]))
        x = torch.cat(pieces, dim=1)
        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias], dim=0
            )
            x = x + bias[None]
        return x


class DualProjectionRelationBlock(nn.Module):
    """Directed token relations followed by the unchanged channel mixer."""

    def __init__(
        self,
        *,
        d_token: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str,
        graph_dynamic_rank: int,
        graph_temperature: float,
    ) -> None:
        super().__init__()
        self.graph_norm = nn.LayerNorm(d_token)
        self.head_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.tail_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.graph_out = nn.Linear(d_token, d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation(activation)
        self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.graph_dynamic_rank = int(graph_dynamic_rank)
        self.graph_temperature = float(graph_temperature)

    def forward(self, x: Tensor) -> Tensor:
        x_residual = x
        h = self.graph_norm(x)
        z_h = self.head_proj(h)
        z_t = self.tail_proj(h)
        scores = torch.matmul(z_h, z_t.transpose(1, 2)) / math.sqrt(
            self.graph_dynamic_rank
        )
        attention = torch.softmax(scores / max(self.graph_temperature, 1e-6), dim=-1)
        mixed = torch.matmul(attention, h)
        graph_update = self.dropout(self.graph_out(mixed))
        x = x_residual + self.graph_scale * graph_update

        x_residual = x
        channel = self.channel_norm(x)
        channel = self.channel_down(channel)
        channel = self.activation(channel)
        channel = self.dropout(channel)
        channel = self.channel_up(channel)
        return x_residual + self.channel_scale * channel


class _HingeMixDP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        n_layers: int = 1,
        d_token: int = 1024,
        d_ffn_factor: float = 0.66,
        ffn_dropout: ty.Optional[float] = None,
        residual_dropout: ty.Optional[float] = 0.1,
        num_breakpoints: int = 8,
        learnable_breakpoints: bool = True,
        slimtok_channel_ratio: float = 2.0,
        slimtok_dropout: ty.Optional[float] = None,
        slimtok_layerscale_init: float = 1e-2,
        slimtok_activation: str = "gelu",
        graph_dynamic_rank: int = 16,
        graph_temperature: float = 16.0,
        d_out: int = 1,
        **_: ty.Any,
    ) -> None:
        super().__init__()
        self.tokenizer = GGPLTokenizer(
            d_numerical=d_numerical,
            categories=categories,
            d_token=d_token,
            bias=token_bias,
            num_breakpoints=num_breakpoints,
            learnable_breakpoints=learnable_breakpoints,
        )
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = slimtok_dropout if slimtok_dropout is not None else (
            ffn_dropout if ffn_dropout is not None else residual_dropout
        )
        self.layers = nn.ModuleList(
            [
                DualProjectionRelationBlock(
                    d_token=d_token,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
                    graph_dynamic_rank=graph_dynamic_rank,
                    graph_temperature=graph_temperature,
                )
                for _ in range(n_layers)
            ]
        )
        self.normalization = nn.LayerNorm(d_token)
        self.activation = _activation(slimtok_activation)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        x = self.normalization(x[:, 0])
        return self.head(self.activation(x)).squeeze(-1)


class HingeMixDP(TabModel):
    """Training adapter for HingeMix-DP, independent of GGPL model classes."""

    display_name = "HingeMix-DP"

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
    ) -> None:
        if feat_gate or pruning:
            raise NotImplementedError("hingemix_dp does not support sparse gating options")
        super().__init__()
        config = self.preproc_config(dict(model_config))
        self.model = _HingeMixDP(
            d_numerical=n_num_features, categories=categories, d_out=n_labels, **config
        ).to(device)
        self.base_name = "hingemix_dp"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/hingemix_dp_breakpoints"
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))
        self.n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.n_trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        print(
            f"[hingemix_dp] parameters={self.n_parameters} "
            f"trainable_parameters={self.n_trainable_parameters} "
            f"K={self.num_breakpoints} r={config['graph_dynamic_rank']} "
            f"tau={config['graph_temperature']}"
        )

    def preproc_config(self, model_config: dict) -> dict:
        self.saved_model_config = model_config.copy()
        for key in (
            "model_name", "base_model", "breakpoint_init", "breakpoint_fallback",
            "breakpoint_cache_dir", "ablation", "graph_dynamic_scale_init",
            "graph_self_loop_init", "slimtok_rank_ratio", "slimtok_min_rank", "slimtok_rank",
        ):
            model_config.pop(key, None)
        defaults = {
            "n_layers": 1, "d_token": 1024, "token_bias": True,
            "d_ffn_factor": 0.66, "ffn_dropout": None, "residual_dropout": 0.1,
            "num_breakpoints": 8, "learnable_breakpoints": True,
            "slimtok_channel_ratio": 2.0, "slimtok_dropout": None,
            "slimtok_layerscale_init": 1e-2, "slimtok_activation": "gelu",
            "graph_dynamic_rank": 16, "graph_temperature": 16.0,
        }
        for key, value in defaults.items():
            model_config.setdefault(key, value)
        return model_config

    @staticmethod
    def _quantile_breakpoints(x_np: np.ndarray, n_breakpoints: int) -> np.ndarray:
        return np.quantile(x_np, np.linspace(0.1, 0.9, n_breakpoints), axis=0).T.astype("float32")

    @staticmethod
    def _even_breakpoints(x_np: np.ndarray, n_breakpoints: int) -> np.ndarray:
        mins, maxs = np.nanmin(x_np, axis=0), np.nanmax(x_np, axis=0)
        qs = np.linspace(0.1, 0.9, n_breakpoints)
        return np.stack([mins + q * (maxs - mins) for q in qs], axis=1).astype("float32")

    def _cache_file(self, save_path: str, n_features: int) -> Path:
        return Path(self.breakpoint_cache_dir) / Path(save_path).name / (
            f"breakpoints_f{n_features}_b{self.num_breakpoints}.json"
        )

    @staticmethod
    def _extract_gbdt_thresholds(x_np: np.ndarray, y_np: np.ndarray) -> list[list[float]]:
        thresholds = [[] for _ in range(x_np.shape[1])]
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            gbdt = GradientBoostingRegressor(
                n_estimators=64, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            gbdt.fit(x_np, y_np.reshape(-1))
            for estimator in gbdt.estimators_.ravel():
                tree = estimator.tree_
                for feature_idx, threshold in zip(tree.feature, tree.threshold):
                    if feature_idx >= 0 and np.isfinite(threshold):
                        thresholds[int(feature_idx)].append(float(threshold))
        except Exception as error:
            print(f"[hingemix_dp] GBDT threshold extraction failed, falling back: {error}")
        return thresholds

    def _fit_or_load_breakpoints(self, x_num: Tensor, ys: Tensor, save_path: str) -> None:
        if x_num is None or x_num.shape[1] == 0:
            return
        cache_file = self._cache_file(save_path, x_num.shape[1])
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.model.tokenizer.set_breakpoints(
                torch.tensor(payload["breakpoints"], dtype=torch.float32, device=self.device)
            )
            print(f"[hingemix_dp] loaded breakpoints: {cache_file}")
            return
        x_np, y_np = x_num.detach().cpu().numpy(), ys.detach().cpu().numpy()
        fallback = (
            self._quantile_breakpoints(x_np, self.num_breakpoints)
            if self.breakpoint_fallback == "quantile"
            else self._even_breakpoints(x_np, self.num_breakpoints)
        )
        breakpoints = fallback.copy()
        if self.breakpoint_init == "gbdt":
            for feature_idx, values in enumerate(self._extract_gbdt_thresholds(x_np, y_np)):
                unique = np.array(sorted(set(values)), dtype="float32")
                if len(unique) >= self.num_breakpoints:
                    pick = np.linspace(0, len(unique) - 1, self.num_breakpoints).round().astype(int)
                    breakpoints[feature_idx] = unique[pick]
                elif len(unique) > 0:
                    merged = np.array(sorted(set(unique.tolist() + fallback[feature_idx].tolist())), dtype="float32")
                    pick = np.linspace(0, len(merged) - 1, self.num_breakpoints).round().astype(int)
                    breakpoints[feature_idx] = merged[pick]
        elif self.breakpoint_init not in ("quantile", "even"):
            raise ValueError(f"Unsupported breakpoint_init: {self.breakpoint_init}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as file:
            json.dump(
                {"breakpoint_init": self.breakpoint_init, "breakpoint_fallback": self.breakpoint_fallback,
                 "num_breakpoints": self.num_breakpoints, "breakpoints": breakpoints.tolist()},
                file, indent=2,
            )
        self.model.tokenizer.set_breakpoints(
            torch.tensor(breakpoints, dtype=torch.float32, device=self.device)
        )
        print(f"[hingemix_dp] saved breakpoints: {cache_file}")

    @staticmethod
    def _collect_breakpoint_fit_data(
        train_loader: ty.Optional[ty.Tuple[DataLoader, ty.Sequence[str]]],
    ) -> tuple[ty.Optional[Tensor], ty.Optional[Tensor]]:
        if train_loader is None:
            return None, None
        loader, placeholders = train_loader
        x_num_batches, y_batches = [], []
        for batch in loader:
            x_num = y = None
            for index, placeholder in enumerate(placeholders):
                if placeholder == "X_num":
                    x_num = batch[index]
                elif placeholder == "y":
                    y = batch[index]
            if x_num is not None and y is not None:
                x_num_batches.append(x_num.detach().cpu())
                y_batches.append(y.detach().cpu())
        if not x_num_batches or not y_batches:
            return None, None
        return torch.cat(x_num_batches), torch.cat(y_batches)

    def fit(
        self, train_loader=None, X_num=None, X_cat=None, ys=None, ids=None, y_std=None,
        eval_set=None, patience: int = 0, task: str = None, training_args=None,
        meta_args: ty.Optional[dict] = None,
    ):
        if task != "regression":
            raise NotImplementedError("hingemix_dp currently supports regression only")
        meta_args = {} if meta_args is None else meta_args
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args
        bp_x_num, bp_ys = X_num, ys
        if bp_x_num is None or bp_ys is None:
            bp_x_num, bp_ys = self._collect_breakpoint_fit_data(train_loader)
        if bp_x_num is not None and bp_ys is not None:
            self._fit_or_load_breakpoints(bp_x_num, bp_ys, meta_args["save_path"])

        def train_step(model, x_num, x_cat, _y):
            start = time.time()
            return model(x_num, x_cat), time.time() - start

        return self.dnn_fit(
            dnn_fit_func=train_step, train_loader=train_loader, X_num=X_num, X_cat=X_cat,
            ys=ys, ids=ids, y_std=y_std, eval_set=eval_set, patience=patience,
            task=task, training_args=training_args, meta_args=meta_args,
        )

    def predict(
        self, dev_loader=None, X_num=None, X_cat=None, ys=None, ids=None, y_std=None,
        task=None, return_probs=True, return_metric=False, return_loss=False,
        meta_args=None,
    ):
        def inference_step(model, x_num, x_cat):
            start = time.time()
            return model(x_num, x_cat), time.time() - start

        return self.dnn_predict(
            dnn_predict_func=inference_step, dev_loader=dev_loader, X_num=X_num,
            X_cat=X_cat, ys=ys, ids=ids, y_std=y_std, task=task,
            return_probs=return_probs, return_metric=return_metric,
            return_loss=return_loss, meta_args=meta_args,
        )

    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)
