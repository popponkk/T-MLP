# %%
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .abstract import TabModel
from .ggpl_tmlp import GGPLTMLP, GGPLTokenizer
from .ggpl_tmlp_graph_slimtok import GraphSlimTokBlock


def _resolve_activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported slimtok activation: {name}")


class SafeGraphLevelReadout(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_levels: int,
        readout_type: str = "weighted_sum",
        dropout: float = 0.0,
        alpha_init: float = 1e-3,
        bias_to_final: bool = True,
        final_bias_value: float = 4.0,
    ) -> None:
        super().__init__()
        self.d_token = d_token
        self.n_levels = n_levels
        self.readout_type = readout_type
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)

        if readout_type == "weighted_sum":
            self.readout_logits = nn.Parameter(torch.zeros(n_levels))
            if bias_to_final:
                with torch.no_grad():
                    self.readout_logits.fill_(0.0)
                    self.readout_logits[-1] = final_bias_value
        elif readout_type == "concat":
            self.proj = nn.Linear(d_token * n_levels, d_token)
        else:
            raise NotImplementedError(f"Unsupported readout type: {readout_type}")

    def forward(self, cls_list: list[Tensor]) -> Tensor:
        if len(cls_list) != self.n_levels:
            raise ValueError(
                f"Expected {self.n_levels} cls tensors, got {len(cls_list)}"
            )

        final_cls = cls_list[-1]
        if self.readout_type == "weighted_sum":
            cls_stack = torch.stack(cls_list, dim=1)
            weights = torch.softmax(self.readout_logits, dim=0)
            readout_cls = torch.sum(cls_stack * weights.view(1, -1, 1), dim=1)
            readout_cls = self.dropout(readout_cls)
        else:
            concat_cls = torch.cat(cls_list, dim=-1)
            readout_cls = self.proj(concat_cls)
            readout_cls = self.dropout(readout_cls)

        return final_cls + self.alpha * (readout_cls - final_cls)


class _GGPLTMLPSafeGraphReadoutSlimTok(nn.Module):
    """GGPL-TMLP with Graph-SlimTok backbone and safe residual CLS readout."""

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
        graph_dynamic_scale_init: float = 1e-2,
        graph_self_loop_init: float = 2.0,
        graph_readout_type: str = "weighted_sum",
        graph_readout_include_input: bool = True,
        graph_readout_dropout: float = 0.0,
        graph_readout_alpha_init: float = 1e-3,
        graph_readout_bias_to_final: bool = True,
        graph_readout_final_bias_value: float = 4.0,
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
        self.graph_readout_type = graph_readout_type
        self.graph_readout_include_input = graph_readout_include_input
        self.channel_hidden = channel_hidden
        self.layers = nn.ModuleList(
            [
                GraphSlimTokBlock(
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
                )
                for _ in range(n_layers)
            ]
        )

        n_readout_levels = n_layers + 1 if graph_readout_include_input else n_layers
        if n_readout_levels < 1:
            raise ValueError("safe graph readout requires at least one level")
        self.readout = SafeGraphLevelReadout(
            d_token=d_token,
            n_levels=n_readout_levels,
            readout_type=graph_readout_type,
            dropout=graph_readout_dropout,
            alpha_init=graph_readout_alpha_init,
            bias_to_final=graph_readout_bias_to_final,
            final_bias_value=graph_readout_final_bias_value,
        )
        self.activation = _resolve_activation(slimtok_activation)
        self.normalization = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        cls_list = []
        if self.graph_readout_include_input:
            cls_list.append(x[:, 0])

        for layer in self.layers:
            x = layer(x)
            cls_list.append(x[:, 0])

        x = self.readout(cls_list)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPSafeGraphReadoutSlimTok(GGPLTMLP):
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
                "ggpl_tmlp_safe_graph_readout_slimtok keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPSafeGraphReadoutSlimTok(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_safe_graph_readout_slimtok"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir",
            "artifacts/ggpl_tmlp_safe_graph_readout_slimtok_breakpoints",
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
        model_config.setdefault("graph_readout_type", "weighted_sum")
        model_config.setdefault("graph_readout_include_input", True)
        model_config.setdefault("graph_readout_dropout", 0.0)
        model_config.setdefault("graph_readout_alpha_init", 1e-3)
        model_config.setdefault("graph_readout_bias_to_final", True)
        model_config.setdefault("graph_readout_final_bias_value", 4.0)
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
