# %%
import math
import time
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .abstract import TabModel
from .ggpl_tmlp import GGPLTMLP, GGPLTokenizer


def _resolve_activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported butterfly activation: {name}")


class ButterflyTokenMix(nn.Module):
    """Hierarchical stage-shared 2x2 butterfly token mixing."""

    def __init__(
        self,
        n_tokens: int,
        init_offdiag: float = 0.01,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        padded_tokens = 1 << math.ceil(math.log2(max(1, n_tokens)))
        num_stages = int(math.log2(padded_tokens)) if padded_tokens > 1 else 1
        self.n_tokens = n_tokens
        self.padded_tokens = padded_tokens
        self.num_stages = num_stages

        mix_mats = torch.zeros(num_stages, 2, 2)
        if identity_init:
            mix_mats[:, 0, 0] = 1.0
            mix_mats[:, 1, 1] = 1.0
            mix_mats[:, 0, 1] = init_offdiag
            mix_mats[:, 1, 0] = init_offdiag
        else:
            mix_mats.fill_(init_offdiag)
        self.mix_mats = nn.Parameter(mix_mats)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d_token = x.shape
        if self.padded_tokens > n_tokens:
            pad = x.new_zeros(batch_size, self.padded_tokens - n_tokens, d_token)
            x_stage = torch.cat([x, pad], dim=1)
        else:
            x_stage = x

        for stage in range(self.num_stages):
            stride = 2 ** stage
            block = stride * 2
            y_stage = x_stage.clone()
            m00 = self.mix_mats[stage, 0, 0]
            m01 = self.mix_mats[stage, 0, 1]
            m10 = self.mix_mats[stage, 1, 0]
            m11 = self.mix_mats[stage, 1, 1]

            for start in range(0, self.padded_tokens, block):
                for offset in range(stride):
                    i = start + offset
                    j = start + offset + stride
                    if j >= self.padded_tokens:
                        continue
                    xi = x_stage[:, i, :]
                    xj = x_stage[:, j, :]
                    y_stage[:, i, :] = m00 * xi + m01 * xj
                    y_stage[:, j, :] = m10 * xi + m11 * xj
            x_stage = y_stage

        return x_stage[:, :n_tokens, :]


class ButterflyTokBlock(nn.Module):
    """Butterfly token mixing followed by channel mixing."""

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
        init_offdiag: float = 0.01,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.token_mix = ButterflyTokenMix(
            n_tokens=n_tokens,
            init_offdiag=init_offdiag,
            identity_init=identity_init,
        )
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.token_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.token_norm(x)
        h = self.token_mix(h)
        x = x_res + self.token_scale * h

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        x = x_res + self.channel_scale * h
        return x


class _GGPLTMLPButterflyTok(nn.Module):
    """GGPL-TMLP with butterfly-style hierarchical token mixing blocks."""

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
        butterfly_channel_ratio: float = 2.0,
        butterfly_dropout: ty.Optional[float] = None,
        butterfly_activation: str = "gelu",
        butterfly_layerscale_init: float = 1e-2,
        butterfly_init_offdiag: float = 0.01,
        butterfly_identity_init: bool = True,
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
        channel_hidden = max(16, int(d_token * butterfly_channel_ratio))
        dropout = (
            butterfly_dropout
            if butterfly_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )

        self.layers = nn.ModuleList(
            [
                ButterflyTokBlock(
                    n_tokens=n_tokens,
                    d_token=d_token,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=butterfly_layerscale_init,
                    activation=butterfly_activation,
                    init_offdiag=butterfly_init_offdiag,
                    identity_init=butterfly_identity_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.activation = _resolve_activation(butterfly_activation)
        self.normalization = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPButterflyTok(GGPLTMLP):
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
                "ggpl_tmlp_butterflytok keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPButterflyTok(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_butterflytok"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_butterflytok_breakpoints"
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
        model_config.setdefault("butterfly_channel_ratio", 2.0)
        model_config.setdefault("butterfly_dropout", None)
        model_config.setdefault("butterfly_activation", "gelu")
        model_config.setdefault("butterfly_layerscale_init", 1e-2)
        model_config.setdefault("butterfly_init_offdiag", 0.01)
        model_config.setdefault("butterfly_identity_init", True)
        return model_config

    def fit(
        self,
        train_loader: ty.Optional[ty.Tuple[ty.Any, int]] = None,
        X_num: ty.Optional[torch.Tensor] = None,
        X_cat: ty.Optional[torch.Tensor] = None,
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None,
        eval_set: ty.Tuple[torch.Tensor, ty.Any] = None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        return super().fit(
            train_loader=train_loader,
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            ids=ids,
            y_std=y_std,
            eval_set=eval_set,
            patience=patience,
            task=task,
            training_args=training_args,
            meta_args=meta_args,
        )

    def predict(
        self,
        dev_loader: ty.Optional[ty.Tuple[ty.Any, int]] = None,
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
        return super().predict(
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
