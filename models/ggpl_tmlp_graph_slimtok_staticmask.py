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


def _build_topk_mask(logits: Tensor, k: int) -> Tensor:
    """Return a bool mask that keeps top-k entries along the last dim."""
    k = max(1, min(int(k), logits.shape[-1]))
    topk_idx = torch.topk(logits, k=k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter(-1, topk_idx, True)


class StaticMaskedDynamicGraphSlimTokBlock(nn.Module):
    """Static graph provides a structural mask, dynamic graph assigns weights."""

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
        graph_dynamic_rank: int = 16,
        graph_temperature: float = 1.0,
        graph_dynamic_scale_init: float = 1e-2,
        graph_self_loop_init: float = 2.0,
        staticmask_topk: ty.Optional[int] = None,
        debug_staticmask: bool = False,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.graph_dynamic_rank = graph_dynamic_rank
        self.graph_temperature = float(graph_temperature)
        self.staticmask_topk = (
            max(1, min(int(staticmask_topk), n_tokens))
            if staticmask_topk is not None
            else max(1, min(math.ceil(math.sqrt(n_tokens)), n_tokens))
        )
        self.debug_staticmask = bool(debug_staticmask)
        self._debug_printed = False

        self.graph_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.graph_logits = nn.Parameter(torch.empty(n_tokens, n_tokens))
        self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
        self.graph_out = nn.Linear(d_token, d_token)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

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
        static_logits = self.graph_logits

        static_mask = _build_topk_mask(static_logits, self.staticmask_topk)
        static_mask = static_mask.unsqueeze(0).expand(dynamic_logits.shape[0], -1, -1)

        n_tokens = dynamic_logits.shape[-1]
        self_mask = torch.eye(n_tokens, dtype=torch.bool, device=x.device).unsqueeze(0)
        self_mask = self_mask.expand(dynamic_logits.shape[0], -1, -1)

        cls_mask = torch.zeros_like(static_mask)
        cls_mask[:, 0, :] = True
        cls_mask[:, :, 0] = True

        mask = static_mask | self_mask | cls_mask

        if self.debug_staticmask and not self._debug_printed:
            static_edge_density = static_mask.float().mean().item()
            final_mask_density = mask.float().mean().item()
            print(
                "[staticmask] "
                f"n_tokens={n_tokens} "
                f"staticmask_topk={self.staticmask_topk} "
                f"static_edge_density={static_edge_density:.6f} "
                f"final_mask_density={final_mask_density:.6f}"
            )
            self._debug_printed = True

        # Static graph is used only as a structural candidate mask.
        # Dynamic graph still determines the final attention weights.
        # The final attention is computed from static-masked dynamic logits.
        # This directly tests whether the static graph is useful as an edge prior.
        neg_inf = torch.finfo(dynamic_logits.dtype).min
        masked_dynamic_logits = dynamic_logits.masked_fill(~mask, neg_inf)
        temperature = max(self.graph_temperature, 1e-6)
        a = torch.softmax(masked_dynamic_logits / temperature, dim=-1)

        h_graph = torch.matmul(a, h)
        h_graph = self.graph_out(h_graph)
        h_graph = self.dropout(h_graph)
        x = x_res + self.graph_scale * h_graph

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        x = x_res + self.channel_scale * h
        return x


class _GGPLTMLPGraphSlimTokStaticMask(nn.Module):
    """GGPL-TMLP with static-masked dynamic graph token mixing."""

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
        staticmask_topk: ty.Optional[int] = None,
        debug_staticmask: bool = False,
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
        self.graph_dynamic_rank = graph_dynamic_rank
        self.channel_hidden = channel_hidden
        self.layers = nn.ModuleList(
            [
                StaticMaskedDynamicGraphSlimTokBlock(
                    n_tokens=n_tokens,
                    d_token=d_token,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
                    graph_dynamic_rank=graph_dynamic_rank,
                    graph_temperature=graph_temperature,
                    graph_dynamic_scale_init=graph_dynamic_scale_init,
                    graph_self_loop_init=graph_self_loop_init,
                    staticmask_topk=staticmask_topk,
                    debug_staticmask=debug_staticmask,
                )
                for _ in range(n_layers)
            ]
        )
        self.activation = _resolve_activation(slimtok_activation)
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


class GGPLTMLPGraphSlimTokStaticMask(GGPLTMLP):
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
                "ggpl_tmlp_graph_slimtok_staticmask keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPGraphSlimTokStaticMask(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_graph_slimtok_staticmask"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir",
            "artifacts/ggpl_tmlp_graph_slimtok_staticmask_breakpoints",
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
        model_config.setdefault("staticmask_topk", None)
        model_config.setdefault("debug_staticmask", False)
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
