"""Hinge-Mix TokenMLP: token-axis MLP mixing with the formal HingeMix tokenizer."""

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


class TokenMLPChannelBlock(nn.Module):
    """Mix token positions, then apply the unchanged per-token channel MLP."""

    def __init__(
        self,
        *,
        d_token: int,
        n_tokens: int,
        token_mlp_ratio: float,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str,
    ) -> None:
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
        self.activation = _resolve_activation(activation)
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


class _HingeMixTokenMLP(nn.Module):
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
        token_mlp_ratio: float = 2.0,
        slimtok_channel_ratio: float = 2.0,
        slimtok_dropout: ty.Optional[float] = None,
        slimtok_layerscale_init: float = 1e-2,
        slimtok_activation: str = "gelu",
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
                TokenMLPChannelBlock(
                    d_token=d_token,
                    n_tokens=self.tokenizer.n_tokens,
                    token_mlp_ratio=token_mlp_ratio,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
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


class HingeMixTokenMLP(GGPLTMLP):
    """Formal HingeMix tokenizer and training path with token-axis MLP mixing."""

    display_name = "Hinge-Mix-TokenMLP"

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
            raise NotImplementedError("hingemix_token_mlp does not support sparse gating options")
        TabModel.__init__(self)
        config = self.preproc_config(dict(model_config))
        self.model = _HingeMixTokenMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **config,
        ).to(device)
        self.base_name = "hingemix_token_mlp"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/hingemix_token_mlp_breakpoints"
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))
        self.n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.n_trainable_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        self.token_mlp_parameters = sum(
            parameter.numel()
            for layer in self.model.layers
            for parameter in (
                *layer.token_norm.parameters(),
                *layer.token_fc1.parameters(),
                *layer.token_fc2.parameters(),
                layer.token_scale,
            )
        )
        self.channel_mlp_parameters = sum(
            parameter.numel()
            for layer in self.model.layers
            for parameter in (
                *layer.channel_norm.parameters(),
                *layer.channel_down.parameters(),
                *layer.channel_up.parameters(),
                layer.channel_scale,
            )
        )
        print(
            f"[hingemix_token_mlp] parameters={self.n_parameters} "
            f"trainable_parameters={self.n_trainable_parameters}"
        )
        print(
            f"[hingemix_token_mlp] token_mlp_parameters={self.token_mlp_parameters} "
            f"channel_mlp_parameters={self.channel_mlp_parameters}"
        )

    def preproc_config(self, model_config: dict) -> dict:
        # Graph-only settings are deliberately ignored rather than silently becoming model parameters.
        for key in (
            "graph_dynamic_rank",
            "graph_temperature",
            "graph_dynamic_scale_init",
            "graph_self_loop_init",
            "slimtok_rank_ratio",
            "slimtok_min_rank",
            "slimtok_rank",
        ):
            model_config.pop(key, None)
        self.saved_model_config = model_config.copy()
        for key in (
            "model_name",
            "base_model",
            "breakpoint_init",
            "breakpoint_fallback",
            "breakpoint_cache_dir",
            "ablation",
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
            "token_mlp_ratio": 2.0,
            "slimtok_channel_ratio": 2.0,
            "slimtok_dropout": None,
            "slimtok_layerscale_init": 1e-2,
            "slimtok_activation": "gelu",
        }
        for key, value in defaults.items():
            model_config.setdefault(key, value)
        return model_config
