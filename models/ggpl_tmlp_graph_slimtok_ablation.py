# %%
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


class GraphSlimTokNoGraphBlock(nn.Module):
    """Channel mixing only ablation block."""

    def __init__(
        self,
        d_token: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.channel_norm = nn.LayerNorm(d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        h = self.dropout(h)
        x = x_res + self.channel_scale * h
        return x


class GraphSlimTokNoChannelBlock(nn.Module):
    """Graph token mixing only ablation block."""

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        dropout: float,
        layerscale_init: float,
        graph_dynamic_rank: int = 16,
        graph_temperature: float = 1.0,
        graph_dynamic_scale_init: float = 1e-2,
        graph_self_loop_init: float = 2.0,
    ) -> None:
        super().__init__()
        self.graph_norm = nn.LayerNorm(d_token)
        self.graph_logits = nn.Parameter(torch.empty(n_tokens, n_tokens))
        self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.graph_dynamic_scale = nn.Parameter(
            torch.ones(1) * graph_dynamic_scale_init
        )
        self.graph_out = nn.Linear(d_token, d_token)
        self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.dropout = nn.Dropout(dropout)
        self.graph_dynamic_rank = graph_dynamic_rank
        self.graph_temperature = float(graph_temperature)

        nn.init.normal_(self.graph_logits, mean=0.0, std=0.02)
        with torch.no_grad():
            self.graph_logits.fill_(0.0)
            self.graph_logits.add_(torch.eye(n_tokens) * graph_self_loop_init)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.graph_norm(x)

        z = self.graph_dynamic_proj(h)
        dynamic_logits = torch.matmul(z, z.transpose(1, 2)) / math.sqrt(
            self.graph_dynamic_rank
        )
        logits = self.graph_logits.unsqueeze(0) + self.graph_dynamic_scale * dynamic_logits
        temperature = max(self.graph_temperature, 1e-6)
        a = torch.softmax(logits / temperature, dim=-1)

        h = torch.matmul(a, h)
        h = self.graph_out(h)
        h = self.dropout(h)
        x = x_res + self.graph_scale * h
        return x


class _GGPLTMLPGraphSlimTokNoGraph(nn.Module):
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
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )
        self.layers = nn.ModuleList(
            [
                GraphSlimTokNoGraphBlock(
                    d_token=d_token,
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

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class _GGPLTMLPGraphSlimTokNoChannel(nn.Module):
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
        graph_dynamic_rank: int = 16,
        graph_temperature: float = 1.0,
        graph_dynamic_scale_init: float = 1e-2,
        graph_self_loop_init: float = 2.0,
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
        n_tokens = self.tokenizer.n_tokens
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )
        self.layers = nn.ModuleList(
            [
                GraphSlimTokNoChannelBlock(
                    n_tokens=n_tokens,
                    d_token=d_token,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    graph_dynamic_rank=graph_dynamic_rank,
                    graph_temperature=graph_temperature,
                    graph_dynamic_scale_init=graph_dynamic_scale_init,
                    graph_self_loop_init=graph_self_loop_init,
                )
                for _ in range(n_layers)
            ]
        )
        self.normalization = nn.LayerNorm(d_token)
        self.activation = _resolve_activation(slimtok_activation)
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


class _GGPLTMLPGraphSlimTokNoBlock(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        d_token: int = 1024,
        num_breakpoints: int = 8,
        learnable_breakpoints: bool = True,
        slimtok_activation: str = "gelu",
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
        self.layers = nn.ModuleList([])
        self.normalization = nn.LayerNorm(d_token)
        self.activation = _resolve_activation(slimtok_activation)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPGraphSlimTokNoGraph(GGPLTMLP):
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
                "ggpl_tmlp_graph_slimtok_no_graph keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPGraphSlimTokNoGraph(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_graph_slimtok_no_graph"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir",
            "artifacts/ggpl_tmlp_graph_slimtok_no_graph_breakpoints",
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
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
        model_config.pop("graph_dynamic_rank", None)
        model_config.pop("graph_temperature", None)
        model_config.pop("graph_dynamic_scale_init", None)
        model_config.pop("graph_self_loop_init", None)
        return model_config


class GGPLTMLPGraphSlimTokNoChannel(GGPLTMLP):
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
                "ggpl_tmlp_graph_slimtok_no_channel keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPGraphSlimTokNoChannel(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_graph_slimtok_no_channel"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir",
            "artifacts/ggpl_tmlp_graph_slimtok_no_channel_breakpoints",
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
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
        model_config.setdefault("graph_dynamic_rank", 16)
        model_config.setdefault("graph_temperature", 1.0)
        model_config.setdefault("graph_dynamic_scale_init", 1e-2)
        model_config.setdefault("graph_self_loop_init", 2.0)
        return model_config


class GGPLTMLPGraphSlimTokNoBlock(GGPLTMLP):
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
                "ggpl_tmlp_graph_slimtok_no_block keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPGraphSlimTokNoBlock(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_graph_slimtok_no_block"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir",
            "artifacts/ggpl_tmlp_graph_slimtok_no_block_breakpoints",
        )
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.pop("slimtok_rank_ratio", None)
        model_config.pop("slimtok_min_rank", None)
        model_config.pop("slimtok_rank", None)
        model_config.pop("slimtok_channel_ratio", None)
        model_config.pop("slimtok_dropout", None)
        model_config.pop("slimtok_layerscale_init", None)
        model_config.pop("graph_dynamic_rank", None)
        model_config.pop("graph_temperature", None)
        model_config.pop("graph_dynamic_scale_init", None)
        model_config.pop("graph_self_loop_init", None)
        model_config.setdefault("d_token", 1024)
        model_config.setdefault("token_bias", True)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("learnable_breakpoints", True)
        model_config.setdefault("slimtok_activation", "gelu")
        return model_config
