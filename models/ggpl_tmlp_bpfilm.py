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


class GGPLTokenizerWithState(GGPLTokenizer):
    """GGPL tokenizer that also returns the numeric piecewise basis state."""

    def forward(
        self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]
    ) -> tuple[Tensor, ty.Optional[Tensor]]:
        x_some = x_num if x_num is not None else x_cat
        assert x_some is not None

        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(len(x_some), 1, -1)
        pieces = [cls]
        basis_state = None

        if self.d_numerical > 0 and x_num is not None:
            breakpoints = self.breakpoints().to(x_num.device)
            basis_state = F.relu(x_num.unsqueeze(-1) - breakpoints.unsqueeze(0))
            basis_state = torch.cat([x_num.unsqueeze(-1), basis_state], dim=-1)
            numeric_tokens = (
                torch.einsum("bdf,dft->bdt", basis_state, self.basis_weight)
                + self.basis_bias
            )
            pieces.append(numeric_tokens)

        if x_cat is not None:
            pieces.append(
                self.category_embeddings(x_cat + self.category_offsets[None])
            )

        x = torch.cat(pieces, dim=1)

        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias], dim=0
            )
            x = x + bias[None]
        return x, basis_state


class BreakpointStateFiLM(nn.Module):
    """Numeric-token-only FiLM modulation conditioned on breakpoint basis state."""

    def __init__(
        self,
        d_token: int,
        basis_dim: int,
        hidden_multiplier: float = 1.0,
        dropout: float = 0.0,
        alpha_init: float = 1e-3,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(d_token * hidden_multiplier))
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_token),
        )
        self.film_head = nn.Linear(d_token, 2 * d_token)
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        if zero_init:
            nn_init.zeros_(self.film_head.weight)
            nn_init.zeros_(self.film_head.bias)

    def forward(self, hidden: Tensor, basis_state: ty.Optional[Tensor]) -> Tensor:
        if basis_state is None:
            return hidden
        d_num = basis_state.shape[1]
        if d_num == 0:
            return hidden

        state = self.state_encoder(basis_state)
        gamma_beta = self.film_head(state)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        h_num = hidden[:, 1 : 1 + d_num, :]
        h_num_new = h_num * (1.0 + self.alpha * gamma) + self.alpha * beta

        out = hidden.clone()
        out[:, 1 : 1 + d_num, :] = h_num_new
        return out


class _GGPLTMLPBPFiLM(nn.Module):
    """Original GGPL-TMLP blocks plus breakpoint-state FiLM after each block."""

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
        bpfilm_alpha_init: float = 1e-3,
        bpfilm_dropout: float = 0.0,
        bpfilm_hidden_multiplier: float = 1.0,
        bpfilm_zero_init: bool = True,
        d_out: int,
    ) -> None:
        super().__init__()
        self.tokenizer = GGPLTokenizerWithState(
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
        basis_dim = num_breakpoints + 1

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
        self.bpfilm_layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "linear0": nn.Linear(d_token, d_hidden * 2),
                    "sgu": SGU(n_tokens),
                    "linear1": nn.Linear(d_hidden, d_token),
                }
            )
            self.layers.append(layer)
            self.bpfilm_layers.append(
                BreakpointStateFiLM(
                    d_token=d_token,
                    basis_dim=basis_dim,
                    hidden_multiplier=bpfilm_hidden_multiplier,
                    dropout=bpfilm_dropout,
                    alpha_init=bpfilm_alpha_init,
                    zero_init=bpfilm_zero_init,
                )
            )

        self.activation = F.gelu
        self.normalization = make_normalization()
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x, basis_state = self.tokenizer(x_num, x_cat)

        for layer, bpfilm in zip(self.layers, self.bpfilm_layers):
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
            x = bpfilm(x, basis_state)

        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPBPFiLM(GGPLTMLP):
    """GGPL-TMLP variant that exposes GGPL basis state for per-layer FiLM modulation."""

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
                "ggpl_tmlp_bpfilm keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPBPFiLM(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_bpfilm"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_bpfilm_breakpoints"
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
        model_config.setdefault("bpfilm_alpha_init", 1e-3)
        model_config.setdefault("bpfilm_dropout", 0.0)
        model_config.setdefault("bpfilm_hidden_multiplier", 1.0)
        model_config.setdefault("bpfilm_zero_init", True)
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
