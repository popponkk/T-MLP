"""Centralized CLS-readout ablations for the formal dynamic-only GGPL-GTM."""

import math
import time
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from .abstract import TabModel, check_dir
from .ggpl_gtm import DynamicOnlyGraphTokenChannelBlock, _resolve_activation
from .ggpl_tmlp import GGPLTMLP, GGPLTokenizer


ABLATIONS = {
    "full": {"tokenizer_type": "ggpl", "use_graph": True, "use_channel": True},
    "no_channel": {"tokenizer_type": "ggpl", "use_graph": True, "use_channel": False},
    "no_graph": {"tokenizer_type": "ggpl", "use_graph": False, "use_channel": True},
    "no_graph_no_channel": {"tokenizer_type": "ggpl", "use_graph": False, "use_channel": False},
    "linear": {"tokenizer_type": "independent_nnlinear", "use_graph": True, "use_channel": True},
    "linear_no_channel": {"tokenizer_type": "independent_nnlinear", "use_graph": True, "use_channel": False},
    "linear_no_graph": {"tokenizer_type": "independent_nnlinear", "use_graph": False, "use_channel": True},
    "linear_no_graph_no_channel": {"tokenizer_type": "independent_nnlinear", "use_graph": False, "use_channel": False},
}


def resolve_ablation(name: ty.Optional[str]) -> tuple[str, dict]:
    name = "full" if name is None else name
    if name not in ABLATIONS:
        raise ValueError(
            f"Unsupported ggpl_gtm ablation '{name}'. Choose one of {sorted(ABLATIONS)}."
        )
    return name, dict(ABLATIONS[name])


class IndependentNNLinearNumericTokenizer(nn.Module):
    """Compare83's per-feature independent ``nn.Linear(1, d_token)`` tokenizer."""

    def __init__(self, d_numerical: int, d_token: int, bias: bool = True) -> None:
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.linears = nn.ModuleList(
            [nn.Linear(1, d_token, bias=bias) for _ in range(self.d_numerical)]
        )
        self.cls_token = nn.Parameter(torch.empty(d_token))
        nn_init.kaiming_uniform_(self.cls_token.unsqueeze(0), a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return 1 + self.d_numerical

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        if x_cat is not None and x_cat.numel() > 0:
            raise NotImplementedError("ggpl_gtm_ablation linear modes support numerical inputs only.")
        if x_num is None:
            raise ValueError("x_num is required")
        numeric_tokens = torch.stack(
            [linear(x_num[:, j : j + 1]) for j, linear in enumerate(self.linears)],
            dim=1,
        )
        cls = self.cls_token.view(1, 1, -1).expand(x_num.shape[0], 1, -1)
        return torch.cat([cls, numeric_tokens], dim=1)


class DynamicOnlyAblationBlock(nn.Module):
    """Dynamic graph/channel branches that are absent rather than masked when disabled."""

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
        use_graph: bool,
        use_channel: bool,
    ) -> None:
        super().__init__()
        self.use_graph = bool(use_graph)
        self.use_channel = bool(use_channel)
        if self.use_graph and self.use_channel:
            # This construction order exactly matches DynamicOnlyGraphTokenChannelBlock.
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
        elif self.use_graph:
            self.graph_norm = nn.LayerNorm(d_token)
            self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
            self.graph_out = nn.Linear(d_token, d_token)
            self.dropout = nn.Dropout(dropout)
            self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
            self.graph_dynamic_rank = int(graph_dynamic_rank)
            self.graph_temperature = float(graph_temperature)
        elif self.use_channel:
            self.channel_norm = nn.LayerNorm(d_token)
            self.channel_down = nn.Linear(d_token, channel_hidden)
            self.channel_up = nn.Linear(channel_hidden, d_token)
            self.dropout = nn.Dropout(dropout)
            self.activation = _resolve_activation(activation)
            self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_graph:
            x_res = x
            h = self.graph_norm(x)
            z = self.graph_dynamic_proj(h)
            dynamic_logits = torch.matmul(z, z.transpose(1, 2)) / math.sqrt(
                self.graph_dynamic_rank
            )
            attention = torch.softmax(
                dynamic_logits / max(self.graph_temperature, 1e-6), dim=-1
            )
            h = self.graph_out(torch.matmul(attention, h))
            x = x_res + self.graph_scale * self.dropout(h)
        if self.use_channel:
            x_res = x
            h = self.channel_norm(x)
            h = self.channel_down(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.channel_up(h)
            x = x_res + self.channel_scale * h
        return x


class _GGPLGTMAblation(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        tokenizer_type: str,
        use_graph: bool,
        use_channel: bool,
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
        if tokenizer_type == "ggpl":
            self.tokenizer = GGPLTokenizer(
                d_numerical=d_numerical,
                categories=categories,
                d_token=d_token,
                bias=token_bias,
                num_breakpoints=num_breakpoints,
                learnable_breakpoints=learnable_breakpoints,
            )
        elif tokenizer_type == "independent_nnlinear":
            if categories:
                raise NotImplementedError(
                    "ggpl_gtm_ablation linear modes support numerical datasets only."
                )
            self.tokenizer = IndependentNNLinearNumericTokenizer(
                d_numerical=d_numerical, d_token=d_token, bias=token_bias
            )
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )
        block_cls = (
            DynamicOnlyGraphTokenChannelBlock if use_graph and use_channel else DynamicOnlyAblationBlock
        )
        block_kwargs = {
            "d_token": d_token,
            "channel_hidden": channel_hidden,
            "dropout": dropout or 0.0,
            "layerscale_init": slimtok_layerscale_init,
            "activation": slimtok_activation,
            "graph_dynamic_rank": graph_dynamic_rank,
            "graph_temperature": graph_temperature,
        }
        if block_cls is DynamicOnlyAblationBlock:
            block_kwargs.update(use_graph=use_graph, use_channel=use_channel)
        self.layers = nn.ModuleList([block_cls(**block_kwargs) for _ in range(n_layers)])
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


class GGPLGTMAblation(GGPLTMLP):
    """One public entry point for all eight formal dynamic-only ablations."""

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
                "ggpl_gtm_ablation does not support sparse gating options"
            )
        raw_config = dict(model_config)
        self.ablation, spec = resolve_ablation(raw_config.pop("ablation", "full"))
        TabModel.__init__(self)
        config = self.preproc_config(raw_config, use_ggpl_tokenizer=spec["tokenizer_type"] == "ggpl")
        self.saved_model_config["ablation"] = self.ablation
        self.model = _GGPLGTMAblation(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            tokenizer_type=spec["tokenizer_type"],
            use_graph=spec["use_graph"],
            use_channel=spec["use_channel"],
            **config,
        ).to(device)
        self.base_name = f"ggpl_gtm_ablation/{self.ablation}"
        self.device = torch.device(device)
        self.use_ggpl_tokenizer = spec["tokenizer_type"] == "ggpl"
        if self.use_ggpl_tokenizer:
            self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
            self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
            self.breakpoint_cache_dir = self.saved_model_config.get(
                "breakpoint_cache_dir", f"artifacts/ggpl_gtm_ablation/{self.ablation}_breakpoints"
            )
            self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))
        else:
            self.breakpoint_init = None
            self.breakpoint_fallback = None
            self.breakpoint_cache_dir = None
            self.num_breakpoints = 0
        n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            f"[ggpl_gtm_ablation/{self.ablation}] tokenizer={spec['tokenizer_type']} "
            f"graph={spec['use_graph']} channel={spec['use_channel']} "
            f"readout=cls parameters={n_parameters}"
        )

    def preproc_config(self, model_config: dict, *, use_ggpl_tokenizer: bool) -> dict:
        self.saved_model_config = model_config.copy()
        for key in (
            "model_name", "base_model", "ablation", "graph_dynamic_scale_init",
            "graph_self_loop_init", "slimtok_rank_ratio", "slimtok_min_rank", "slimtok_rank",
        ):
            model_config.pop(key, None)
        if use_ggpl_tokenizer:
            for key in ("breakpoint_init", "breakpoint_fallback", "breakpoint_cache_dir"):
                model_config.pop(key, None)
            model_config.setdefault("num_breakpoints", 8)
            model_config.setdefault("learnable_breakpoints", True)
        else:
            for key in (
                "breakpoint_init", "breakpoint_fallback", "breakpoint_cache_dir",
                "num_breakpoints", "learnable_breakpoints", "gbdt", "gbdt_params",
            ):
                model_config.pop(key, None)
        defaults = {
            "n_layers": 1,
            "d_token": 1024,
            "token_bias": True,
            "d_ffn_factor": 0.66,
            "ffn_dropout": None,
            "residual_dropout": 0.1,
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

    def fit(
        self,
        train_loader=None,
        X_num=None,
        X_cat=None,
        ys=None,
        ids=None,
        y_std=None,
        eval_set=None,
        patience: int = 0,
        task: str = None,
        training_args=None,
        meta_args=None,
    ):
        if self.use_ggpl_tokenizer:
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
        if task != "regression":
            raise NotImplementedError("ggpl_gtm_ablation supports regression only")
        meta_args = {} if meta_args is None else meta_args
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        def train_step(model, x_num, x_cat, y):
            start_time = time.time()
            return model(x_num, x_cat), time.time() - start_time

        return self.dnn_fit(
            dnn_fit_func=train_step,
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
