# %%
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


class LIRSlimTokBlock(nn.Module):
    """Low-rank token mixing with latent interaction refinement."""

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        token_rank: int,
        channel_hidden: int,
        dropout: float,
        layerscale_init: float,
        activation: str = "gelu",
        lir_hidden_ratio: float = 1.0,
        lir_layerscale_init: float = 1e-2,
        lir_use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_token)
        self.channel_norm = nn.LayerNorm(d_token)
        self.token_down = nn.Linear(n_tokens, token_rank)
        self.token_up = nn.Linear(token_rank, n_tokens)
        self.lir_use_norm = lir_use_norm
        self.latent_norm = nn.LayerNorm(token_rank) if lir_use_norm else nn.Identity()
        lir_hidden = max(1, int(token_rank * lir_hidden_ratio))
        self.lir_hidden = lir_hidden
        self.latent_down = nn.Linear(token_rank, lir_hidden)
        self.latent_up = nn.Linear(lir_hidden, token_rank)
        self.latent_scale = nn.Parameter(torch.ones(1) * lir_layerscale_init)
        self.channel_down = nn.Linear(d_token, channel_hidden)
        self.channel_up = nn.Linear(channel_hidden, d_token)
        self.dropout = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.token_scale = nn.Parameter(torch.ones(1) * layerscale_init)
        self.channel_scale = nn.Parameter(torch.ones(1) * layerscale_init)

    def forward(self, x: Tensor) -> Tensor:
        x_res = x
        h = self.token_norm(x)
        h = h.transpose(1, 2)

        z = self.token_down(h)
        z = self.activation(z)
        z = self.dropout(z)

        z_res = z
        u = self.latent_norm(z)
        u = self.latent_down(u)
        u = self.activation(u)
        u = self.dropout(u)
        u = self.latent_up(u)
        z = z_res + self.latent_scale * u

        h = self.token_up(z)
        h = h.transpose(1, 2)
        x = x_res + self.token_scale * h

        x_res = x
        h = self.channel_norm(x)
        h = self.channel_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.channel_up(h)
        x = x_res + self.channel_scale * h
        return x


class _GGPLTMLPLIRSlimTok(nn.Module):
    """GGPL-TMLP with latent interaction refinement in low-rank token space."""

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
        slimtok_rank_ratio: float = 0.25,
        slimtok_min_rank: int = 4,
        slimtok_rank: ty.Optional[int] = None,
        slimtok_channel_ratio: float = 2.0,
        slimtok_dropout: ty.Optional[float] = None,
        slimtok_layerscale_init: float = 1e-2,
        slimtok_activation: str = "gelu",
        lir_hidden_ratio: float = 1.0,
        lir_layerscale_init: float = 1e-2,
        lir_use_norm: bool = True,
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
        token_rank = (
            slimtok_rank
            if slimtok_rank is not None
            else max(slimtok_min_rank, int(n_tokens * slimtok_rank_ratio))
        )
        token_rank = max(1, min(token_rank, n_tokens))
        channel_hidden = max(1, int(d_token * slimtok_channel_ratio))
        dropout = (
            slimtok_dropout
            if slimtok_dropout is not None
            else (ffn_dropout if ffn_dropout is not None else residual_dropout)
        )

        self.n_tokens = n_tokens
        self.token_rank = token_rank
        self.channel_hidden = channel_hidden
        self.layers = nn.ModuleList(
            [
                LIRSlimTokBlock(
                    n_tokens=n_tokens,
                    d_token=d_token,
                    token_rank=token_rank,
                    channel_hidden=channel_hidden,
                    dropout=dropout or 0.0,
                    layerscale_init=slimtok_layerscale_init,
                    activation=slimtok_activation,
                    lir_hidden_ratio=lir_hidden_ratio,
                    lir_layerscale_init=lir_layerscale_init,
                    lir_use_norm=lir_use_norm,
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


class GGPLTMLPLIRSlimTok(GGPLTMLP):
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
                "ggpl_tmlp_lir_slimtok keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPLIRSlimTok(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_lir_slimtok"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_lir_slimtok_breakpoints"
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
        model_config.setdefault("slimtok_rank_ratio", 0.25)
        model_config.setdefault("slimtok_min_rank", 4)
        model_config.setdefault("slimtok_rank", None)
        model_config.setdefault("slimtok_channel_ratio", 2.0)
        model_config.setdefault("slimtok_dropout", None)
        model_config.setdefault("slimtok_layerscale_init", 1e-2)
        model_config.setdefault("slimtok_activation", "gelu")
        model_config.setdefault("lir_hidden_ratio", 1.0)
        model_config.setdefault("lir_layerscale_init", 1e-2)
        model_config.setdefault("lir_use_norm", True)
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
