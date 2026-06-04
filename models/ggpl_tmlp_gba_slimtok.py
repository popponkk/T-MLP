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
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .ggpl_tmlp import GGPLTMLP, GGPLTokenizer


def _resolve_activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported slimtok activation: {name}")


class GBASlimTokBlock(nn.Module):
    """GBDT-biased self-attention followed by channel mixing."""

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        n_heads: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
        attn_bias_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads})"
            )
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.attn_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.q_proj = nn.Linear(d_token, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out_proj = nn.Linear(d_token, d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.attn_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.gbdt_bias_scale = nn.Parameter(torch.ones(1) * attn_bias_scale_init)
        self.register_buffer(
            "gbdt_attention_bias",
            torch.zeros(n_tokens, n_tokens),
            persistent=False,
        )

    def set_gbdt_attention_bias(self, bias: torch.Tensor) -> None:
        if bias.shape != (self.n_tokens, self.n_tokens):
            raise ValueError(
                f"Expected bias shape {(self.n_tokens, self.n_tokens)}, got {tuple(bias.shape)}"
            )
        self.gbdt_attention_bias.copy_(
            bias.detach().float().to(self.gbdt_attention_bias.device)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d_token = x.shape

        x_res = x
        h = self.attn_norm(x)

        q = self.q_proj(h).view(batch_size, n_tokens, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(batch_size, n_tokens, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(batch_size, n_tokens, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.gbdt_attention_bias.to(logits.device)
        logits = logits + self.gbdt_bias_scale * bias.view(1, 1, n_tokens, n_tokens)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        h = torch.matmul(attn, v)
        h = h.permute(0, 2, 1, 3).contiguous().view(batch_size, n_tokens, d_token)
        h = self.out_proj(h)
        h = self.dropout(h)
        x = x_res + self.attn_scale * h

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        x = x_res + self.channel_scale * h
        return x


class _GGPLTMLPGBASlimTok(nn.Module):
    """GGPL-TMLP with GBDT-biased self-attention token mixing."""

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
        slimtok_channel_ratio: float = 2.0,
        slimtok_dropout: ty.Optional[float] = None,
        slimtok_layerscale_init: float = 1e-2,
        slimtok_activation: str = "gelu",
        gba_n_heads: int = 4,
        gba_attn_bias_scale_init: float = 0.1,
        d_out: int,
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
        self.n_categories = 0 if categories is None else len(categories)
        n_tokens = self.tokenizer.n_tokens
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )

        self.n_tokens = n_tokens
        self.d_token = d_token
        self.n_heads = gba_n_heads
        self.channel_hidden = channel_hidden
        self.layers = nn.ModuleList(
            [
                GBASlimTokBlock(
                    n_tokens=n_tokens,
                    d_token=d_token,
                    n_heads=gba_n_heads,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
                    attn_bias_scale_init=gba_attn_bias_scale_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.activation = _resolve_activation(slimtok_activation)
        self.normalization = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)

    def set_gbdt_attention_bias(self, bias: torch.Tensor) -> None:
        for layer in self.layers:
            layer.set_gbdt_attention_bias(bias)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPGBASlimTok(GGPLTMLP):
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
                "ggpl_tmlp_gba_slimtok keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPGBASlimTok(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_gba_slimtok"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_gba_slimtok_breakpoints"
        )
        self.gba_attention_bias_cache_dir = self.saved_model_config.get(
            "gba_attention_bias_cache_dir",
            "artifacts/ggpl_tmlp_gba_slimtok_attention_bias",
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.pop("gba_attention_bias_cache_dir", None)
        model_config.pop("slimtok_rank_ratio", None)
        model_config.pop("slimtok_min_rank", None)
        model_config.pop("slimtok_rank", None)
        model_config.setdefault("n_layers", 1)
        model_config.setdefault("d_token", 1024)
        model_config.setdefault("token_bias", True)
        model_config.setdefault("d_ffn_factor", 0.66)
        model_config.setdefault("ffn_dropout", None)
        model_config.setdefault("residual_dropout", 0.1)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("learnable_breakpoints", True)
        model_config.setdefault("slimtok_channel_ratio", 2.0)
        model_config.setdefault("slimtok_dropout", None)
        model_config.setdefault("slimtok_layerscale_init", 1e-2)
        model_config.setdefault("slimtok_activation", "gelu")
        model_config.setdefault("gba_n_heads", 4)
        model_config.setdefault("gba_attn_bias_scale_init", 0.1)
        return model_config

    def _attention_bias_cache_file(self, save_path: str, n_features: int) -> Path:
        dataset_name = Path(save_path).name
        return (
            Path(self.gba_attention_bias_cache_dir)
            / dataset_name
            / f"attention_bias_f{n_features}_b{self.num_breakpoints}.json"
        )

    @staticmethod
    def _accumulate_path_cooccurrence(tree, node_id: int, path: list[int], cooc: np.ndarray):
        feature_idx = int(tree.feature[node_id])
        if feature_idx < 0:
            if not path:
                return
            used = sorted(set(path))
            for a in used:
                cooc[a, a] += 1.0
                for b in used:
                    if a != b:
                        cooc[a, b] += 1.0
            return

        next_path = path + [feature_idx]
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])
        if left_id >= 0:
            GGPLTMLPGBASlimTok._accumulate_path_cooccurrence(
                tree, left_id, next_path, cooc
            )
        if right_id >= 0:
            GGPLTMLPGBASlimTok._accumulate_path_cooccurrence(
                tree, right_id, next_path, cooc
            )

    def _build_gbdt_attention_bias(
        self, x_np: np.ndarray, y_np: np.ndarray, n_tokens: int
    ) -> np.ndarray:
        n_features = x_np.shape[1]
        bias = np.zeros((n_tokens, n_tokens), dtype=np.float32)
        if n_features == 0:
            return bias
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
            cooc = np.zeros((n_features, n_features), dtype=np.float32)
            for estimator in gbdt.estimators_.ravel():
                self._accumulate_path_cooccurrence(estimator.tree_, 0, [], cooc)
            if float(cooc.max()) <= 0.0:
                return bias
            cooc = cooc + cooc.T
            cooc = cooc / (float(cooc.max()) + 1e-12)
            np.fill_diagonal(cooc, 0.0)
            bias[1 : 1 + n_features, 1 : 1 + n_features] = cooc
            return bias
        except Exception as err:
            print(f"[ggpl_tmlp_gba_slimtok] GBDT attention bias failed, using zeros: {err}")
            return bias

    def _fit_or_load_gbdt_attention_bias(
        self,
        x_num: torch.Tensor,
        ys: torch.Tensor,
        save_path: str,
    ) -> None:
        n_tokens = self.model.n_tokens
        n_features = 0 if x_num is None else int(x_num.shape[1])
        zero_bias = torch.zeros(n_tokens, n_tokens, dtype=torch.float32, device=self.device)
        if x_num is None or n_features == 0 or ys is None:
            self.model.set_gbdt_attention_bias(zero_bias)
            return

        cache_file = self._attention_bias_cache_file(save_path, n_features)
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                bias = torch.tensor(payload["bias"], dtype=torch.float32, device=self.device)
                self.model.set_gbdt_attention_bias(bias)
                print(f"[ggpl_tmlp_gba_slimtok] loaded attention bias: {cache_file}")
                return
            except Exception as err:
                print(f"[ggpl_tmlp_gba_slimtok] failed to load attention bias cache, rebuilding: {err}")

        x_np = x_num.detach().cpu().numpy()
        y_np = ys.detach().cpu().numpy()
        bias_np = self._build_gbdt_attention_bias(x_np, y_np, n_tokens)
        bias = torch.tensor(bias_np, dtype=torch.float32, device=self.device)
        self.model.set_gbdt_attention_bias(bias)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "gbdt_path_cooccurrence",
                    "n_features": n_features,
                    "n_tokens": n_tokens,
                    "bias": bias_np.tolist(),
                },
                f,
                indent=2,
            )
        print(f"[ggpl_tmlp_gba_slimtok] saved attention bias: {cache_file}")

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
            raise NotImplementedError("ggpl_tmlp_gba_slimtok currently supports regression only")
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
        self._fit_or_load_gbdt_attention_bias(bp_x_num, bp_ys, meta_args["save_path"])

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
