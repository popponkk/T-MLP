# %%
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
    raise ValueError(f"Unsupported setmix activation: {name}")


class SetMixMLP(nn.Module):
    def __init__(self, d_token: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * d_token)
        self.linear0 = nn.Linear(2 * d_token, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, d_token)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.linear0(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x


class SetMixBlock(nn.Module):
    """Global set-context token update followed by channel mixing."""

    def __init__(
        self,
        d_token: int,
        setmix_hidden: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool != "mean":
            raise NotImplementedError(f"Unsupported setmix pool: {pool}")

        self.pool = pool
        self.token_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.setmix_mlp = SetMixMLP(d_token, setmix_hidden, dropout)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.setmix_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.token_norm(x)
        context = h.mean(dim=1, keepdim=True)
        context = context.expand(-1, h.shape[1], -1)
        z = torch.cat([h, context], dim=-1)
        delta = self.setmix_mlp(z)
        x = x_res + self.setmix_scale * delta

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        x = x_res + self.channel_scale * h
        return x


class _GGPLTMLPSetMix(nn.Module):
    """GGPL-TMLP with SetMix backbone blocks."""

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
        setmix_hidden_ratio: float = 2.0,
        setmix_hidden_dim: ty.Optional[int] = None,
        setmix_channel_ratio: float = 2.0,
        setmix_dropout: ty.Optional[float] = None,
        setmix_activation: str = "gelu",
        setmix_layerscale_init: float = 1e-2,
        setmix_pool: str = "mean",
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

        setmix_hidden = (
            setmix_hidden_dim
            if setmix_hidden_dim is not None
            else int(d_token * setmix_hidden_ratio)
        )
        setmix_hidden = max(16, setmix_hidden)
        channel_hidden = max(16, int(d_token * setmix_channel_ratio))
        dropout = (
            setmix_dropout
            if setmix_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )

        self.layers = nn.ModuleList(
            [
                SetMixBlock(
                    d_token=d_token,
                    setmix_hidden=setmix_hidden,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=setmix_layerscale_init,
                    activation=setmix_activation,
                    pool=setmix_pool,
                )
                for _ in range(n_layers)
            ]
        )
        self.activation = _resolve_activation(setmix_activation)
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


class GGPLTMLPSetMix(GGPLTMLP):
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
                "ggpl_tmlp_setmix keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPSetMix(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_setmix"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_setmix_breakpoints"
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
        model_config.setdefault("setmix_hidden_ratio", 2.0)
        model_config.setdefault("setmix_hidden_dim", None)
        model_config.setdefault("setmix_channel_ratio", 2.0)
        model_config.setdefault("setmix_dropout", None)
        model_config.setdefault("setmix_activation", "gelu")
        model_config.setdefault("setmix_layerscale_init", 1e-2)
        model_config.setdefault("setmix_pool", "mean")
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
