"""CLS-readout factorial ablations of the dynamic-only Graph-SlimTok model."""

import math
import time
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

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


class LinearNumericTokenizer(nn.Module):
    """[CLS] + per-feature linear numeric tokens for pure numerical inputs."""

    def __init__(self, d_numerical: int, d_token: int, bias: bool) -> None:
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.weight = nn.Parameter(torch.empty(d_numerical, d_token))
        self.bias = nn.Parameter(torch.empty(d_numerical, d_token)) if bias else None
        self.cls_token = nn.Parameter(torch.empty(d_token))
        nn_init.kaiming_uniform_(self.cls_token.unsqueeze(0), a=math.sqrt(5))
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return 1 + self.d_numerical

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        if x_cat is not None and x_cat.numel() > 0:
            raise NotImplementedError(
                "ggpl_dynonly_ablation supports pure numerical inputs only."
            )
        if x_num is None:
            raise ValueError("LinearNumericTokenizer requires x_num")
        numeric_tokens = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            numeric_tokens = numeric_tokens + self.bias.unsqueeze(0)
        cls = self.cls_token.view(1, 1, -1).expand(x_num.shape[0], 1, -1)
        return torch.cat([cls, numeric_tokens], dim=1)


class DynamicOnlyAblationBlock(nn.Module):
    """Optional dynamic graph and channel branches with no replacement path."""

    def __init__(
        self,
        *,
        n_tokens: int,
        d_token: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str,
        graph_dynamic_rank: int,
        graph_temperature: float,
        graph_self_loop_init: float,
        use_graph: bool,
        use_channel: bool,
    ) -> None:
        super().__init__()
        self.use_graph = bool(use_graph)
        self.use_channel = bool(use_channel)

        if self.use_graph:
            self.graph_norm = nn.LayerNorm(d_token)
            if self.use_channel:
                self.channel_norm = nn.LayerNorm(d_token)
                # The legacy dynamic_only block creates this parameter but
                # never consumes it in forward. Retaining it only here keeps
                # the full ablation state dict and initialization identical.
                self.graph_logits = nn.Parameter(torch.empty(n_tokens, n_tokens))
            self.graph_dynamic_proj = nn.Linear(d_token, graph_dynamic_rank, bias=False)
            self.graph_out = nn.Linear(d_token, d_token)
            if self.use_channel:
                self.channel_down = nn.Linear(d_token, channel_hidden)
                self.channel_up = nn.Linear(channel_hidden, d_token)
            self.dropout = nn.Dropout(dropout)
            self.activation = _resolve_activation(activation) if self.use_channel else None
            self.graph_scale = nn.Parameter(torch.ones(1) * layerscale_init)
            self.channel_scale = (
                nn.Parameter(torch.ones(1) * layerscale_init)
                if self.use_channel
                else None
            )
            self.graph_dynamic_scale = None
            self.graph_dynamic_rank = int(graph_dynamic_rank)
            self.graph_temperature = float(graph_temperature)
            if self.use_channel:
                nn.init.normal_(self.graph_logits, mean=0.0, std=0.02)
                with torch.no_grad():
                    self.graph_logits.fill_(0.0)
                    self.graph_logits.add_(
                        torch.eye(n_tokens) * graph_self_loop_init
                    )
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
            h = torch.matmul(attention, h)
            h = self.graph_out(h)
            h = self.dropout(h)
            x = x_res + self.graph_scale * h

        if self.use_channel:
            x_res = x
            h = self.channel_norm(x)
            h = self.channel_down(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.channel_up(h)
            x = x_res + self.channel_scale * h
        return x


class _GGPLDynOnlyAblationModel(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        use_ggpl_tokenizer: bool,
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
        if categories:
            raise NotImplementedError(
                "ggpl_dynonly_ablation supports pure numerical datasets only; "
                "categorical input is not silently ignored."
            )
        self.use_ggpl_tokenizer = bool(use_ggpl_tokenizer)
        self.use_graph = bool(use_graph)
        self.use_channel = bool(use_channel)
        if self.use_ggpl_tokenizer:
            self.tokenizer = GGPLTokenizer(
                d_numerical=d_numerical,
                categories=None,
                d_token=d_token,
                bias=token_bias,
                num_breakpoints=num_breakpoints,
                learnable_breakpoints=learnable_breakpoints,
            )
        else:
            self.tokenizer = LinearNumericTokenizer(d_numerical, d_token, token_bias)

        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )
        self.layers = nn.ModuleList(
            [
                DynamicOnlyAblationBlock(
                    n_tokens=self.tokenizer.n_tokens,
                    d_token=d_token,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
                    graph_dynamic_rank=graph_dynamic_rank,
                    graph_temperature=graph_temperature,
                    graph_self_loop_init=graph_self_loop_init,
                    use_graph=self.use_graph,
                    use_channel=self.use_channel,
                )
                for _ in range(n_layers)
            ]
        )
        self.normalization = nn.LayerNorm(d_token)
        self.activation = _resolve_activation(slimtok_activation)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor] = None) -> Tensor:
        if x_cat is not None and x_cat.numel() > 0:
            raise NotImplementedError(
                "ggpl_dynonly_ablation supports pure numerical inputs only."
            )
        x = self.tokenizer(x_num, None)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        return self.head(x).squeeze(-1)


class _BaseGGPLDynOnlyAblation(GGPLTMLP):
    model_name: str
    use_ggpl_tokenizer: bool
    use_graph: bool
    use_channel: bool

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
                f"{self.model_name} does not support sparse gating options"
            )
        if categories:
            raise NotImplementedError(
                f"{self.model_name} supports pure numerical datasets only."
            )
        TabModel.__init__(self)
        config = self.preproc_config(dict(model_config))
        self.model = _GGPLDynOnlyAblationModel(
            d_numerical=n_num_features,
            categories=None,
            d_out=n_labels,
            use_ggpl_tokenizer=self.use_ggpl_tokenizer,
            use_graph=self.use_graph,
            use_channel=self.use_channel,
            **config,
        ).to(device)
        self.base_name = self.model_name
        self.device = torch.device(device)
        if self.use_ggpl_tokenizer:
            self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
            self.breakpoint_fallback = self.saved_model_config.get(
                "breakpoint_fallback", "quantile"
            )
            self.breakpoint_cache_dir = self.saved_model_config.get(
                "breakpoint_cache_dir", f"artifacts/{self.model_name}_breakpoints"
            )
            self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))
        else:
            self.breakpoint_init = None
            self.breakpoint_fallback = None
            self.breakpoint_cache_dir = None
            self.num_breakpoints = 0
        n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            f"[{self.model_name}] tokenizer={'ggpl' if self.use_ggpl_tokenizer else 'linear'} "
            f"graph={self.use_graph} channel={self.use_channel} "
            f"readout=cls parameters={n_parameters}"
        )

    def preproc_config(self, model_config: dict) -> dict:
        self.saved_model_config = model_config.copy()
        for key in (
            "model_name", "base_model", "slimtok_rank_ratio", "slimtok_min_rank", "slimtok_rank"
        ):
            model_config.pop(key, None)
        if self.use_ggpl_tokenizer:
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
        model_config.setdefault("n_layers", 1)
        model_config.setdefault("d_token", 1024)
        model_config.setdefault("token_bias", True)
        model_config.setdefault("d_ffn_factor", 0.66)
        model_config.setdefault("ffn_dropout", None)
        model_config.setdefault("residual_dropout", 0.1)
        model_config.setdefault("slimtok_channel_ratio", 2.0)
        model_config.setdefault("slimtok_dropout", None)
        model_config.setdefault("slimtok_layerscale_init", 1e-2)
        model_config.setdefault("slimtok_activation", "gelu")
        model_config.setdefault("graph_dynamic_rank", 16)
        model_config.setdefault("graph_temperature", 1.0)
        model_config.setdefault("graph_dynamic_scale_init", 1e-2)
        model_config.setdefault("graph_self_loop_init", 2.0)
        return model_config

    def fit(self, train_loader=None, X_num=None, X_cat=None, ys=None, ids=None,
            y_std=None, eval_set=None, patience: int = 0, task: str = None,
            training_args=None, meta_args=None):
        if self.use_ggpl_tokenizer:
            return super().fit(
                train_loader=train_loader, X_num=X_num, X_cat=X_cat, ys=ys,
                ids=ids, y_std=y_std, eval_set=eval_set, patience=patience,
                task=task, training_args=training_args, meta_args=meta_args,
            )
        if task != "regression":
            raise NotImplementedError("ggpl_dynonly_ablation supports regression only")
        meta_args = {} if meta_args is None else meta_args
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        def train_step(model, x_num, x_cat, y):
            start_time = time.time()
            logits = model(x_num, x_cat)
            return logits, time.time() - start_time

        return self.dnn_fit(
            dnn_fit_func=train_step, train_loader=train_loader, X_num=X_num,
            X_cat=X_cat, ys=ys, y_std=y_std, ids=ids, eval_set=eval_set,
            patience=patience, task=task, training_args=training_args,
            meta_args=meta_args,
        )


class GGPLDynOnlyAblation(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation"
    use_ggpl_tokenizer = True
    use_graph = True
    use_channel = True
