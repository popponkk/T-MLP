# %%
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


class GGPLTokenizer(nn.Module):
    """GGPL numeric tokenizer + original TMLP categorical tokenizer + [CLS] token.

    This module keeps the original TMLP token sequence contract:
    [CLS] + numeric tokens + categorical tokens.

    The only changed part is numeric token generation:
    - original TMLP: per-feature linear scaling token
    - GGPL-TMLP: per-feature piecewise-linear basis -> token projection
    """

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
            pieces.append(
                self.category_embeddings(x_cat + self.category_offsets[None])
            )

        x = torch.cat(pieces, dim=1)

        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias], dim=0
            )
            x = x + bias[None]
        return x


class _GGPLTMLP(nn.Module):
    """Original TMLP backbone with a GGPL numeric tokenizer."""

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
        num_breakpoints: int = 8,
        learnable_breakpoints: bool = True,
        d_out: int,
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
        self.n_categories = 0 if categories is None else len(categories)
        n_tokens = self.tokenizer.n_tokens

        def make_normalization(d=d_token):
            return nn.LayerNorm(d)

        d_hidden = int(d_token * d_ffn_factor)
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
                    "linear0": nn.Linear(d_token, d_hidden * 2),
                    "sgu": SGU(n_tokens),
                    "linear1": nn.Linear(d_hidden, d_token),
                }
            )
            self.layers.append(layer)

        self.activation = F.gelu
        self.normalization = make_normalization()
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        # This backbone is copied from the original TMLP unchanged.
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = x
            x = self.normalization(x)
            x = layer["linear0"](x)
            x = self.activation(x)
            x = layer["sgu"](x)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            x = layer["linear1"](x)
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            x = x + x_residual

        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLP(TabModel):
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
            raise NotImplementedError(
                "ggpl_tmlp keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_breakpoints"
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.setdefault("n_layers", 1)
        model_config.setdefault("d_token", 1024)
        model_config.setdefault("token_bias", True)
        model_config.setdefault("d_ffn_factor", 0.66)
        model_config.setdefault("ffn_dropout", None)
        model_config.setdefault("residual_dropout", 0.1)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("learnable_breakpoints", True)
        return model_config

    @staticmethod
    def _quantile_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        qs = np.linspace(0.1, 0.9, num_breakpoints)
        return np.quantile(x_np, qs, axis=0).T.astype("float32")

    @staticmethod
    def _even_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        mins = np.nanmin(x_np, axis=0)
        maxs = np.nanmax(x_np, axis=0)
        qs = np.linspace(0.1, 0.9, num_breakpoints)
        return np.stack([mins + q * (maxs - mins) for q in qs], axis=1).astype(
            "float32"
        )

    def _cache_file(self, save_path: str, n_features: int) -> Path:
        dataset_name = Path(save_path).name
        return (
            Path(self.breakpoint_cache_dir)
            / dataset_name
            / f"breakpoints_f{n_features}_b{self.num_breakpoints}.json"
        )

    def _extract_gbdt_thresholds(
        self, x_np: np.ndarray, y_np: np.ndarray
    ) -> list[list[float]]:
        thresholds = [[] for _ in range(x_np.shape[1])]
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            gbdt = GradientBoostingRegressor(
                n_estimators=64,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            gbdt.fit(x_np, y_np.reshape(-1))
            for estimator in gbdt.estimators_.ravel():
                tree = estimator.tree_
                for feature_idx, threshold in zip(tree.feature, tree.threshold):
                    if feature_idx >= 0 and np.isfinite(threshold):
                        thresholds[int(feature_idx)].append(float(threshold))
        except Exception as err:
            print(f"[ggpl_tmlp] GBDT threshold extraction failed, falling back: {err}")
        return thresholds

    def _fit_or_load_breakpoints(
        self, x_num: torch.Tensor, ys: torch.Tensor, save_path: str
    ) -> None:
        if x_num is None or x_num.shape[1] == 0:
            return
        cache_file = self._cache_file(save_path, x_num.shape[1])
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            breakpoints = torch.tensor(
                payload["breakpoints"], dtype=torch.float32, device=self.device
            )
            self.model.tokenizer.set_breakpoints(breakpoints)
            print(f"[ggpl_tmlp] loaded breakpoints: {cache_file}")
            return

        x_np = x_num.detach().cpu().numpy()
        y_np = ys.detach().cpu().numpy()
        fallback = (
            self._quantile_breakpoints(x_np, self.num_breakpoints)
            if self.breakpoint_fallback == "quantile"
            else self._even_breakpoints(x_np, self.num_breakpoints)
        )
        breakpoints = fallback.copy()

        if self.breakpoint_init == "gbdt":
            thresholds = self._extract_gbdt_thresholds(x_np, y_np)
            for feature_idx, values in enumerate(thresholds):
                unique = np.array(sorted(set(values)), dtype="float32")
                if len(unique) >= self.num_breakpoints:
                    pick = (
                        np.linspace(0, len(unique) - 1, self.num_breakpoints)
                        .round()
                        .astype(int)
                    )
                    breakpoints[feature_idx] = unique[pick]
                elif len(unique) > 0:
                    merged = np.array(
                        sorted(set(unique.tolist() + fallback[feature_idx].tolist())),
                        dtype="float32",
                    )
                    pick = (
                        np.linspace(0, len(merged) - 1, self.num_breakpoints)
                        .round()
                        .astype(int)
                    )
                    breakpoints[feature_idx] = merged[pick]
        elif self.breakpoint_init not in ["quantile", "even"]:
            raise ValueError(f"Unsupported breakpoint_init: {self.breakpoint_init}")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "breakpoint_init": self.breakpoint_init,
                    "breakpoint_fallback": self.breakpoint_fallback,
                    "num_breakpoints": self.num_breakpoints,
                    "breakpoints": breakpoints.tolist(),
                },
                f,
                indent=2,
            )
        self.model.tokenizer.set_breakpoints(
            torch.tensor(breakpoints, dtype=torch.float32, device=self.device)
        )
        print(f"[ggpl_tmlp] saved breakpoints: {cache_file}")

    @staticmethod
    def _collect_breakpoint_fit_data(
        train_loader: ty.Optional[ty.Tuple[DataLoader, ty.Sequence[str]]],
    ) -> tuple[ty.Optional[torch.Tensor], ty.Optional[torch.Tensor]]:
        if train_loader is None:
            return None, None
        loader, placeholders = train_loader
        x_num_batches = []
        y_batches = []
        for batch in loader:
            x_num = None
            y = None
            for idx, ph in enumerate(placeholders):
                if ph == "X_num":
                    x_num = batch[idx]
                elif ph == "y":
                    y = batch[idx]
            if x_num is not None and y is not None:
                x_num_batches.append(x_num.detach().cpu())
                y_batches.append(y.detach().cpu())
        if not x_num_batches or not y_batches:
            return None, None
        return torch.cat(x_num_batches, dim=0), torch.cat(y_batches, dim=0)

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
            raise NotImplementedError("ggpl_tmlp currently supports regression only")
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        bp_x_num = X_num
        bp_ys = ys
        if bp_x_num is None or bp_ys is None:
            bp_x_num, bp_ys = self._collect_breakpoint_fit_data(train_loader)
        if bp_x_num is not None and bp_ys is not None:
            self._fit_or_load_breakpoints(bp_x_num, bp_ys, meta_args["save_path"])

        def train_step(model, x_num, x_cat, y):
            start_time = time.time()
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time
            return logits, used_time

        self.dnn_fit(
            dnn_fit_func=train_step,
            train_loader=train_loader,
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            y_std=y_std,
            ids=ids,
            eval_set=eval_set,
            patience=patience,
            task=task,
            training_args=training_args,
            meta_args=meta_args,
        )

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
