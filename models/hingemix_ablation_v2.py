"""Self-contained four-way HingeMix ablation experiment.

Production code in this module depends only on the project's generic TabModel
training interface.  It intentionally does not import another model module.
"""

import hashlib
import json
import math
import os
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


ABLATIONS = {
    "full": {"tokenizer_type": "ggpl", "mixing_type": "graph"},
    "token_mlp": {"tokenizer_type": "ggpl", "mixing_type": "token_mlp"},
    "linear": {"tokenizer_type": "shared_linear", "mixing_type": "graph"},
    "linear_token_mlp": {"tokenizer_type": "shared_linear", "mixing_type": "token_mlp"},
}


def resolve_ablation(name: ty.Optional[str]) -> tuple[str, dict[str, str]]:
    name = "full" if name is None else name
    if name not in ABLATIONS:
        raise ValueError(
            f"Unsupported hingemix_ablation_v2 mode {name!r}. "
            f"Choose one of {sorted(ABLATIONS)}; legacy no_graph modes are not aliases."
        )
    return name, dict(ABLATIONS[name])


def resolve_activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported slimtok activation: {name}")


class PiecewiseLinearTokenizer(nn.Module):
    """Formal GGPL tokenizer, copied here to keep this experiment self-contained."""

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
            offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.cls_token = nn.Parameter(Tensor(d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        if self.d_numerical:
            self.basis_weight = nn.Parameter(
                torch.empty(self.d_numerical, self.num_breakpoints + 1, d_token)
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
        return 1 + self.d_numerical + (0 if self.category_offsets is None else len(self.category_offsets))

    @staticmethod
    def _inverse_softplus(x: Tensor) -> Tensor:
        return torch.log(torch.expm1(torch.clamp(x, min=1e-6)))

    def set_breakpoints(self, breakpoints: Tensor) -> None:
        if not self.d_numerical:
            return
        breakpoints = torch.sort(breakpoints.detach().float(), dim=1).values
        first = breakpoints[:, :1]
        deltas = torch.diff(breakpoints, dim=1, prepend=first)
        deltas[:, :1] = 1e-3
        deltas = torch.clamp(deltas, min=1e-4)
        self.breakpoint_min = (first - deltas[:, :1]).to(self.breakpoint_min.device)
        self.breakpoint_delta_raw.data.copy_(self._inverse_softplus(deltas).to(self.breakpoint_delta_raw.device))

    def breakpoints(self) -> Tensor:
        if not self.d_numerical:
            return torch.zeros(0, self.num_breakpoints, device=self.cls_token.device)
        deltas = F.softplus(self.breakpoint_delta_raw)
        return self.breakpoint_min.to(deltas.device) + torch.cumsum(deltas, dim=1)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_num is not None else x_cat
        if x_some is None:
            raise ValueError("At least one numerical or categorical input is required")
        pieces = [self.cls_token.view(1, 1, -1).expand(len(x_some), 1, -1)]
        if self.d_numerical and x_num is not None:
            points = self.breakpoints().to(x_num.device)
            basis = F.relu(x_num.unsqueeze(-1) - points.unsqueeze(0))
            basis = torch.cat([x_num.unsqueeze(-1), basis], dim=-1)
            pieces.append(torch.einsum("bdf,dft->bdt", basis, self.basis_weight) + self.basis_bias)
        if x_cat is not None:
            pieces.append(self.category_embeddings(x_cat + self.category_offsets[None]))
        x = torch.cat(pieces, dim=1)
        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias], dim=0)
            x = x + bias[None]
        return x


class SharedLinearTokenizer(nn.Module):
    """CLS plus one truly shared scalar-to-token linear mapping."""

    def __init__(self, d_numerical: int, d_token: int, bias: bool = True) -> None:
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.linear = nn.Linear(1, d_token, bias=bias)
        self.cls_token = nn.Parameter(torch.empty(d_token))
        nn_init.kaiming_uniform_(self.cls_token.unsqueeze(0), a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return self.d_numerical + 1

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        if x_cat is not None and x_cat.numel():
            raise NotImplementedError("Shared Linear modes support numerical inputs only")
        if x_num is None or x_num.ndim != 2 or x_num.shape[1] != self.d_numerical:
            got = None if x_num is None else tuple(x_num.shape)
            raise ValueError(f"Expected x_num [B, {self.d_numerical}], got {got}")
        tokens = self.linear(x_num.unsqueeze(-1))
        cls = self.cls_token.view(1, 1, -1).expand(x_num.shape[0], 1, -1)
        return torch.cat([cls, tokens], dim=1)


class GraphChannelBlock(nn.Module):
    """Single-projection input-adaptive relation mixing followed by channel mixing."""

    def __init__(self, *, d_token: int, channel_hidden: int, dropout: float,
                 layerscale_init: float, activation: str, graph_dynamic_rank: int,
                 graph_temperature: float) -> None:
        super().__init__()
        # Construction order matches formal HingeMix for direct state-dict transfer.
        self.graph_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.graph_out = nn.Linear(d_token, d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = resolve_activation(activation)
        self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.graph_dynamic_rank = int(graph_dynamic_rank)
        self.graph_temperature = float(graph_temperature)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.graph_norm(x)
        z = self.graph_dynamic_proj(h)
        logits = torch.matmul(z, z.transpose(1, 2)) / math.sqrt(self.graph_dynamic_rank)
        attention = torch.softmax(logits / max(self.graph_temperature, 1e-6), dim=-1)
        h = self.dropout(self.graph_out(torch.matmul(attention, h)))
        x = residual + self.graph_scale * h
        residual = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        return residual + self.channel_scale * h


class TokenMLPChannelBlock(nn.Module):
    """Token-axis MLP replacement followed by the unchanged channel branch."""

    def __init__(self, *, d_token: int, n_tokens: int, token_mlp_ratio: float,
                 channel_hidden: int, dropout: float, layerscale_init: float,
                 activation: str) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.hidden_tokens = max(1, int(self.n_tokens * token_mlp_ratio))
        self.token_norm = nn.LayerNorm(d_token)
        self.token_fc1 = nn.Linear(self.n_tokens, self.hidden_tokens, bias=True)
        self.token_fc2 = nn.Linear(self.hidden_tokens, self.n_tokens, bias=True)
        self.channel_norm = nn.LayerNorm(d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = resolve_activation(activation)
        self.token_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.token_norm(x).transpose(1, 2)
        h = self.token_fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.token_fc2(h).transpose(1, 2)
        x = residual + self.token_scale * h
        residual = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        return residual + self.channel_scale * h


class _HingeMixAblationV2Net(nn.Module):
    def __init__(self, *, d_numerical: int, categories: ty.Optional[ty.List[int]],
                 tokenizer_type: str, mixing_type: str, token_bias: bool,
                 n_layers: int = 1, d_token: int = 1024, d_ffn_factor: float = 0.66,
                 ffn_dropout: ty.Optional[float] = None, residual_dropout: ty.Optional[float] = 0.1,
                 num_breakpoints: int = 8, learnable_breakpoints: bool = True,
                 graph_dynamic_rank: int = 16, graph_temperature: float = 16.0,
                 token_mlp_ratio: float = 2.0, slimtok_channel_ratio: float = 2.0,
                 slimtok_dropout: ty.Optional[float] = None, slimtok_layerscale_init: float = 1e-2,
                 slimtok_activation: str = "gelu", d_out: int = 1, **_: ty.Any) -> None:
        super().__init__()
        if tokenizer_type == "ggpl":
            self.tokenizer = PiecewiseLinearTokenizer(
                d_numerical, categories, d_token, token_bias, num_breakpoints, learnable_breakpoints
            )
        elif tokenizer_type == "shared_linear":
            if categories:
                raise NotImplementedError("Shared Linear modes support numerical datasets only")
            self.tokenizer = SharedLinearTokenizer(d_numerical, d_token, token_bias)
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = slimtok_dropout if slimtok_dropout is not None else (
            ffn_dropout if ffn_dropout is not None else residual_dropout
        )
        if mixing_type == "graph":
            block = lambda: GraphChannelBlock(
                d_token=d_token, channel_hidden=channel_hidden, dropout=dropout or 0.0,
                layerscale_init=slimtok_layerscale_init, activation=slimtok_activation,
                graph_dynamic_rank=graph_dynamic_rank, graph_temperature=graph_temperature,
            )
        elif mixing_type == "token_mlp":
            block = lambda: TokenMLPChannelBlock(
                d_token=d_token, n_tokens=self.tokenizer.n_tokens, token_mlp_ratio=token_mlp_ratio,
                channel_hidden=channel_hidden, dropout=dropout or 0.0,
                layerscale_init=slimtok_layerscale_init, activation=slimtok_activation,
            )
        else:
            raise ValueError(f"Unsupported mixing_type: {mixing_type}")
        self.layers = nn.ModuleList([block() for _ in range(n_layers)])
        self.normalization = nn.LayerNorm(d_token)
        self.activation = resolve_activation(slimtok_activation)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.activation(self.normalization(x[:, 0]))).squeeze(-1)


class HingeMixAblationV2(TabModel):
    """Four formal v2 ablations with local tokenizer and breakpoint training logic."""

    display_name = "HingeMix ablation v2"

    def __init__(self, model_config: dict, n_num_features: int,
                 categories: ty.Optional[ty.List[int]], n_labels: int,
                 device: ty.Union[str, torch.device] = "cuda", feat_gate=None,
                 pruning=None, dataset=None) -> None:
        if feat_gate or pruning:
            raise NotImplementedError("hingemix_ablation_v2 does not support sparse gating options")
        raw = dict(model_config)
        self.ablation, spec = resolve_ablation(raw.pop("ablation", "full"))
        super().__init__()
        config = self.preproc_config(raw, spec)
        self.model = _HingeMixAblationV2Net(
            d_numerical=n_num_features, categories=categories, d_out=n_labels,
            tokenizer_type=spec["tokenizer_type"], mixing_type=spec["mixing_type"], **config,
        ).to(device)
        self.base_name = f"hingemix_ablation_v2/{self.ablation}"
        self.device = torch.device(device)
        self.use_ggpl_tokenizer = spec["tokenizer_type"] == "ggpl"
        self.tokenizer_type = spec["tokenizer_type"]
        self.mixing_type = spec["mixing_type"]
        if self.use_ggpl_tokenizer:
            self.breakpoint_init = self.saved_model_config["breakpoint_init"]
            self.breakpoint_fallback = self.saved_model_config["breakpoint_fallback"]
            self.breakpoint_cache_dir = self.saved_model_config["breakpoint_cache_dir"]
            self.num_breakpoints = int(self.saved_model_config["num_breakpoints"])
        else:
            self.breakpoint_init = self.breakpoint_fallback = self.breakpoint_cache_dir = None
            self.num_breakpoints = 0
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        self.n_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.token_mlp_parameters = sum(
            p.numel() for layer in self.model.layers if isinstance(layer, TokenMLPChannelBlock)
            for p in (*layer.token_norm.parameters(), *layer.token_fc1.parameters(), *layer.token_fc2.parameters(), layer.token_scale)
        )
        self.channel_mlp_parameters = sum(
            p.numel() for layer in self.model.layers
            for p in (*layer.channel_norm.parameters(), *layer.channel_down.parameters(), *layer.channel_up.parameters(), layer.channel_scale)
        )
        print(f"[hingemix_ablation_v2/{self.ablation}] tokenizer={self.tokenizer_type} "
              f"mixing={self.mixing_type} channel=True readout=cls parameters={self.n_parameters} "
              f"trainable_parameters={self.n_trainable_parameters}")
        print(f"[hingemix_ablation_v2/{self.ablation}] token_mlp_parameters={self.token_mlp_parameters} "
              f"channel_mlp_parameters={self.channel_mlp_parameters}")

    def preproc_config(self, model_config: dict, spec: dict[str, str]) -> dict:
        requested = dict(model_config)
        for key in ("model_name", "base_model", "ablation", "graph_dynamic_scale_init",
                    "graph_self_loop_init", "slimtok_rank_ratio", "slimtok_min_rank", "slimtok_rank"):
            model_config.pop(key, None)
        is_ggpl, is_graph = spec["tokenizer_type"] == "ggpl", spec["mixing_type"] == "graph"
        metadata = {"ablation": self.ablation, "tokenizer_type": spec["tokenizer_type"],
                    "mixing_type": spec["mixing_type"], "channel_mixing": True,
                    "cls_readout": True, "requested_model_config": requested}
        if is_ggpl:
            breakpoint_init = model_config.pop("breakpoint_init", "gbdt")
            breakpoint_fallback = model_config.pop("breakpoint_fallback", "quantile")
            breakpoint_cache_dir = model_config.pop(
                "breakpoint_cache_dir", f"artifacts/hingemix_ablation_v2_breakpoints/{self.ablation}"
            )
            metadata.update(breakpoint_init=breakpoint_init, breakpoint_fallback=breakpoint_fallback,
                            breakpoint_cache_dir=breakpoint_cache_dir)
        else:
            for key in ("breakpoint_init", "breakpoint_fallback", "breakpoint_cache_dir", "num_breakpoints",
                        "learnable_breakpoints", "gbdt", "gbdt_params"):
                model_config.pop(key, None)
            metadata["non_applicable_parameters"] = [
                "num_breakpoints", "learnable_breakpoints", "breakpoint_init", "breakpoint_fallback", "breakpoint_cache_dir"
            ]
        if not is_graph:
            model_config.pop("graph_dynamic_rank", None)
            model_config.pop("graph_temperature", None)
            metadata.setdefault("non_applicable_parameters", []).extend(["graph_dynamic_rank", "graph_temperature"])
        defaults = {"n_layers": 1, "d_token": 1024, "token_bias": True, "d_ffn_factor": 0.66,
                    "ffn_dropout": None, "residual_dropout": 0.1, "slimtok_channel_ratio": 2.0,
                    "slimtok_dropout": None, "slimtok_layerscale_init": 1e-2, "slimtok_activation": "gelu"}
        if is_ggpl:
            defaults.update(num_breakpoints=8, learnable_breakpoints=True)
        if is_graph:
            defaults.update(graph_dynamic_rank=16, graph_temperature=16.0)
        else:
            defaults.update(token_mlp_ratio=2.0)
        for key, value in defaults.items():
            model_config.setdefault(key, value)
        metadata["effective_model_config"] = dict(model_config)
        self.saved_model_config = metadata
        return model_config

    @staticmethod
    def _quantile_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        return np.quantile(x_np, np.linspace(0.1, 0.9, num_breakpoints), axis=0).T.astype("float32")

    @staticmethod
    def _even_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        mins, maxs = np.nanmin(x_np, axis=0), np.nanmax(x_np, axis=0)
        return np.stack([mins + q * (maxs - mins) for q in np.linspace(0.1, 0.9, num_breakpoints)], axis=1).astype("float32")

    @staticmethod
    def _array_digest(value: torch.Tensor) -> str:
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _cache_file(self, save_path: str, n_features: int) -> Path:
        # save_path is task-specific; this namespace cannot overwrite formal caches.
        return Path(self.breakpoint_cache_dir) / Path(save_path).name / f"breakpoints_f{n_features}_b{self.num_breakpoints}.json"

    def _extract_gbdt_thresholds(self, x_np: np.ndarray, y_np: np.ndarray) -> list[list[float]]:
        thresholds = [[] for _ in range(x_np.shape[1])]
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            gbdt = GradientBoostingRegressor(n_estimators=64, max_depth=3, learning_rate=0.05,
                                             subsample=0.8, random_state=42)
            gbdt.fit(x_np, y_np.reshape(-1))
            for estimator in gbdt.estimators_.ravel():
                tree = estimator.tree_
                for feature_idx, threshold in zip(tree.feature, tree.threshold):
                    if feature_idx >= 0 and np.isfinite(threshold):
                        thresholds[int(feature_idx)].append(float(threshold))
        except Exception as error:
            print(f"[hingemix_ablation_v2] GBDT threshold extraction failed; using fallback: {error}")
        return thresholds

    def _fit_or_load_breakpoints(self, x_num: torch.Tensor, ys: torch.Tensor, save_path: str) -> None:
        if x_num is None or not x_num.shape[1]:
            return
        identity = {"schema": 1, "features": int(x_num.shape[1]), "num_breakpoints": self.num_breakpoints,
                    "breakpoint_init": self.breakpoint_init, "breakpoint_fallback": self.breakpoint_fallback,
                    "gbdt": {"library": "sklearn.ensemble.GradientBoostingRegressor", "n_estimators": 64,
                             "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": 42},
                    "train_x_sha256": self._array_digest(x_num), "train_y_sha256": self._array_digest(ys)}
        cache_file = self._cache_file(save_path, x_num.shape[1])
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                if payload.get("identity") == identity:
                    self.model.tokenizer.set_breakpoints(torch.tensor(payload["breakpoints"], dtype=torch.float32, device=self.device))
                    print(f"[hingemix_ablation_v2] loaded breakpoints: {cache_file}")
                    return
                print(f"[hingemix_ablation_v2] rejected incompatible breakpoint cache: {cache_file}")
            except (OSError, ValueError, KeyError) as error:
                print(f"[hingemix_ablation_v2] rejected unreadable breakpoint cache: {error}")
        x_np, y_np = x_num.detach().cpu().numpy(), ys.detach().cpu().numpy()
        fallback = self._quantile_breakpoints(x_np, self.num_breakpoints) if self.breakpoint_fallback == "quantile" else self._even_breakpoints(x_np, self.num_breakpoints)
        breakpoints = fallback.copy()
        if self.breakpoint_init == "gbdt":
            for index, values in enumerate(self._extract_gbdt_thresholds(x_np, y_np)):
                unique = np.array(sorted(set(values)), dtype="float32")
                if len(unique):
                    values = unique if len(unique) >= self.num_breakpoints else np.array(
                        sorted(set(unique.tolist() + fallback[index].tolist())), dtype="float32"
                    )
                    breakpoints[index] = values[np.linspace(0, len(values) - 1, self.num_breakpoints).round().astype(int)]
        elif self.breakpoint_init not in ("quantile", "even"):
            raise ValueError(f"Unsupported breakpoint_init: {self.breakpoint_init}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"identity": identity, "breakpoints": breakpoints.tolist()}
        temporary = cache_file.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, cache_file)
        self.model.tokenizer.set_breakpoints(torch.tensor(breakpoints, dtype=torch.float32, device=self.device))
        print(f"[hingemix_ablation_v2] saved breakpoints: {cache_file}")

    @staticmethod
    def _collect_breakpoint_fit_data(train_loader: ty.Optional[ty.Tuple[DataLoader, ty.Sequence[str]]]):
        if train_loader is None:
            return None, None
        loader, placeholders = train_loader
        x_batches, y_batches = [], []
        for batch in loader:
            x_num = y = None
            for index, placeholder in enumerate(placeholders):
                if placeholder == "X_num":
                    x_num = batch[index]
                elif placeholder == "y":
                    y = batch[index]
            if x_num is not None and y is not None:
                x_batches.append(x_num.detach().cpu())
                y_batches.append(y.detach().cpu())
        return (torch.cat(x_batches), torch.cat(y_batches)) if x_batches and y_batches else (None, None)

    def fit(self, train_loader=None, X_num=None, X_cat=None, ys=None, ids=None, y_std=None,
            eval_set=None, patience: int = 0, task: str = None, training_args=None, meta_args=None):
        if task != "regression":
            raise NotImplementedError("hingemix_ablation_v2 supports regression only")
        meta_args = {} if meta_args is None else meta_args
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args
        if self.use_ggpl_tokenizer:
            bp_x, bp_y = (X_num, ys) if X_num is not None and ys is not None else self._collect_breakpoint_fit_data(train_loader)
            if bp_x is not None and bp_y is not None:
                self._fit_or_load_breakpoints(bp_x, bp_y, meta_args["save_path"])
        def train_step(model, x_num, x_cat, y):
            started = time.time()
            return model(x_num, x_cat), time.time() - started
        return self.dnn_fit(dnn_fit_func=train_step, train_loader=train_loader, X_num=X_num, X_cat=X_cat,
                            ys=ys, ids=ids, y_std=y_std, eval_set=eval_set, patience=patience,
                            task=task, training_args=training_args, meta_args=meta_args)

    def predict(self, dev_loader=None, X_num=None, X_cat=None, ys=None, ids=None, y_std=None,
                task=None, return_probs=True, return_metric=False, return_loss=False, meta_args=None):
        def inference_step(model, x_num, x_cat):
            started = time.time()
            return model(x_num, x_cat), time.time() - started
        return self.dnn_predict(dnn_predict_func=inference_step, dev_loader=dev_loader, X_num=X_num,
                                X_cat=X_cat, ys=ys, y_std=y_std, ids=ids, task=task,
                                return_probs=return_probs, return_metric=return_metric,
                                return_loss=return_loss, meta_args=meta_args)

    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)
