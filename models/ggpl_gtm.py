"""Formal GGPL-GTM: GGPL tokens, dynamic graph mixing, channel mixing, CLS head."""

import math
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
    raise ValueError(f"Unsupported slimtok activation: {name}")


class DynamicOnlyGraphTokenChannelBlock(nn.Module):
    """The static-free dynamic graph and channel-mixing backbone block."""

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
        self.channel_norm = nn.LayerNorm(d_token)
        self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.graph_out = nn.Linear(d_token, d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.graph_dynamic_rank = int(graph_dynamic_rank)
        self.graph_temperature = float(graph_temperature)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.graph_norm(x)
        z = self.graph_dynamic_proj(h)
        dynamic_logits = torch.matmul(z, z.transpose(1, 2)) / math.sqrt(
            self.graph_dynamic_rank
        )
        attention = torch.softmax(
            dynamic_logits / max(self.graph_temperature, 1e-6), dim=-1
        )
        h = torch.matmul(attention, h)
        h = self.graph_out(h)
        h = self.dropout(h)
        x = x_res + self.graph_scale * h

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        return x_res + self.channel_scale * h


class _GGPLGTM(nn.Module):
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
        graph_temperature: float = 1.0,
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
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )
        self.layers = nn.ModuleList(
            [
                DynamicOnlyGraphTokenChannelBlock(
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
        self.activation = _resolve_activation(slimtok_activation)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        return self.head(x).squeeze(-1)


class GGPLGTM(GGPLTMLP):
    """Public fixed GGPL-GTM model with no ablation controls."""

    display_name = "GGPL-GTM"

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
            raise NotImplementedError(
                "ggpl_gtm keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        config = self.preproc_config(dict(model_config))
        self.model = _GGPLGTM(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **config,
        ).to(device)
        self.base_name = "ggpl_gtm"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_gtm_breakpoints"
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict) -> dict:
        self.saved_model_config = model_config.copy()
        for key in (
            "model_name", "base_model", "breakpoint_init", "breakpoint_fallback",
            "breakpoint_cache_dir", "ablation", "graph_dynamic_scale_init",
            "graph_self_loop_init", "slimtok_rank_ratio", "slimtok_min_rank", "slimtok_rank",
        ):
            model_config.pop(key, None)
        defaults = {
            "n_layers": 1,
            "d_token": 1024,
            "token_bias": True,
            "d_ffn_factor": 0.66,
            "ffn_dropout": None,
            "residual_dropout": 0.1,
            "num_breakpoints": 8,
            "learnable_breakpoints": True,
            "slimtok_channel_ratio": 2.0,
            "slimtok_dropout": None,
            "slimtok_layerscale_init": 1e-2,
            "slimtok_activation": "gelu",
            "graph_dynamic_rank": 16,
            "graph_temperature": 1.0,
        }
        for key, value in defaults.items():
            model_config.setdefault(key, value)
        return model_config
