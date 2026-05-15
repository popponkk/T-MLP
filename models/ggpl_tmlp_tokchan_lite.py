# %%
import time
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from .abstract import TabModel
from .ggpl_tmlp import GGPLTMLP, GGPLTokenizer


class ChannelMLP(nn.Module):
    def __init__(self, d_token: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_token)
        self.linear0 = nn.Linear(d_token, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, d_token)
        self._init_weights()

    def _init_weights(self) -> None:
        nn_init.zeros_(self.linear1.weight)
        nn_init.zeros_(self.linear1.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.linear0(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x


class LiteTokChanAdapter(nn.Module):
    def __init__(
        self,
        d_token: int,
        adapter_ratio: float = 0.25,
        adapter_dropout: float = 0.0,
        adapter_alpha_init: float = 1e-3,
        use_cls_gate: bool = True,
        use_token_gate: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(16, int(d_token * adapter_ratio))
        self.use_cls_gate = use_cls_gate
        self.use_token_gate = use_token_gate

        self.token_norm = nn.LayerNorm(d_token)
        self.cls_norm = nn.LayerNorm(d_token)
        self.token_proj = nn.Linear(d_token, d_token)
        self.cls_proj = nn.Linear(d_token, d_token)
        self.beta_head = nn.Linear(d_token, 1)
        self.channel_mlp = ChannelMLP(d_token, hidden, adapter_dropout)
        self.alpha = nn.Parameter(torch.ones(1) * adapter_alpha_init)

        self._init_weights()

    def _init_weights(self) -> None:
        nn_init.zeros_(self.token_proj.bias)
        nn_init.zeros_(self.cls_proj.bias)
        nn_init.zeros_(self.beta_head.bias)

    def forward(self, x: Tensor) -> Tensor:
        cls = x[:, 0, :]
        gate_logits = torch.zeros_like(x)
        if self.use_token_gate:
            gate_logits = gate_logits + self.token_proj(self.token_norm(x))
        if self.use_cls_gate:
            cls_context = self.cls_proj(self.cls_norm(cls)).unsqueeze(1)
            gate_logits = gate_logits + cls_context
        gate = torch.sigmoid(gate_logits)

        beta = torch.sigmoid(self.beta_head(cls)).unsqueeze(1)
        u = gate * x
        delta = self.channel_mlp(u)
        return x + self.alpha * beta * delta


class _GGPLTMLPTokChanLite(nn.Module):
    """Original GGPL-TMLP blocks with a lightweight token-channel residual adapter."""

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
        adapter_ratio: float = 0.25,
        adapter_dropout: float = 0.0,
        adapter_alpha_init: float = 1e-3,
        use_cls_gate: bool = True,
        use_token_gate: bool = True,
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

        def make_normalization(d=d_token):
            return nn.LayerNorm(d)

        d_hidden = int(d_token * d_ffn_factor)
        self.d_ffn = d_hidden

        class SGU(nn.Module):
            def __init__(self, n_token):
                super().__init__()
                self.proj = nn.Linear(n_token, n_token)
                self.norm = make_normalization(d=d_hidden)

            def forward(self, x):
                u, v = torch.chunk(x, 2, -1)
                v = self.norm(v).transpose(1, 2)
                v = self.proj(v).transpose(1, 2)
                return u * v

        self.layers = nn.ModuleList([])
        self.adapters = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "linear0": nn.Linear(d_token, d_hidden * 2),
                    "sgu": SGU(n_tokens),
                    "linear1": nn.Linear(d_hidden, d_token),
                }
            )
            self.layers.append(layer)
            self.adapters.append(
                LiteTokChanAdapter(
                    d_token=d_token,
                    adapter_ratio=adapter_ratio,
                    adapter_dropout=adapter_dropout,
                    adapter_alpha_init=adapter_alpha_init,
                    use_cls_gate=use_cls_gate,
                    use_token_gate=use_token_gate,
                )
            )

        self.activation = F.gelu
        self.normalization = make_normalization()
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer, adapter in zip(self.layers, self.adapters):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = x
            x = self.normalization(x)
            x = layer["linear0"](x)
            x = self.activation(x)
            x = layer["sgu"](x)
            if self.ffn_dropout:
                x = F.dropout(x, self.ffn_dropout, self.training)
            x = layer["linear1"](x)
            if self.residual_dropout:
                x = F.dropout(x, self.residual_dropout, self.training)
            x = x + x_residual
            x = adapter(x)

        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPTokChanLite(GGPLTMLP):
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
                "ggpl_tmlp_tokchan_lite keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPTokChanLite(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_tokchan_lite"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_tokchan_lite_breakpoints"
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
        model_config.setdefault("adapter_ratio", 0.25)
        model_config.setdefault("adapter_dropout", 0.0)
        model_config.setdefault("adapter_alpha_init", 1e-3)
        model_config.setdefault("use_cls_gate", True)
        model_config.setdefault("use_token_gate", True)
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
