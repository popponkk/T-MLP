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
from .ggpl_tmlp_tokchan import ChannelMixing


class GGPLTokenizerWithState(GGPLTokenizer):
    """GGPL tokenizer that also returns numeric piecewise basis states."""

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


class BreakpointGate(nn.Module):
    """Breakpoint-state gate for numeric channel residuals only."""

    def __init__(
        self,
        d_token: int,
        basis_dim: int,
        ratio: float = 0.25,
        dropout: float = 0.0,
        alpha_init: float = 1e-3,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(16, int(d_token * ratio))
        self.gate_net = nn.Sequential(
            nn.LayerNorm(basis_dim),
            nn.Linear(basis_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_token),
        )
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        if zero_init:
            nn_init.zeros_(self.gate_net[-1].weight)
            nn_init.zeros_(self.gate_net[-1].bias)

    def forward(self, basis_state: ty.Optional[Tensor]) -> ty.Optional[Tensor]:
        if basis_state is None:
            return None
        raw_gate = self.gate_net(basis_state)
        return 1.0 + self.alpha * raw_gate


class _GGPLTMLPTokChanBPGate(nn.Module):
    """TokChan backbone whose numeric channel residuals are breakpoint-gated."""

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
        d_channel: ty.Optional[int] = None,
        channel_scale_init: float = 1e-2,
        bp_gate_ratio: float = 0.25,
        bp_gate_dropout: float = 0.0,
        bp_gate_alpha_init: float = 1e-3,
        bp_gate_zero_init: bool = True,
        bp_gate_all_layers: bool = False,
        bp_gate_start_layer: ty.Optional[int] = None,
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
        d_channel = d_channel or max(d_token // 2, 32)
        basis_dim = num_breakpoints + 1
        self.bp_gate_all_layers = bp_gate_all_layers
        self.bp_gate_start_layer = (
            n_layers // 2 if bp_gate_start_layer is None else bp_gate_start_layer
        )

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
        self.bp_gates = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "token_norm": make_normalization(),
                    "linear0": nn.Linear(d_token, d_hidden * 2),
                    "sgu": SGU(n_tokens),
                    "linear1": nn.Linear(d_hidden, d_token),
                    "channel_norm": make_normalization(),
                    "channel_mix": ChannelMixing(d_token, d_channel),
                }
            )
            self.layers.append(layer)
            use_gate = bp_gate_all_layers or layer_idx >= self.bp_gate_start_layer
            self.bp_gates.append(
                BreakpointGate(
                    d_token=d_token,
                    basis_dim=basis_dim,
                    ratio=bp_gate_ratio,
                    dropout=bp_gate_dropout,
                    alpha_init=bp_gate_alpha_init,
                    zero_init=bp_gate_zero_init,
                )
                if use_gate
                else nn.Identity()
            )
        self.channel_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * channel_scale_init) for _ in range(n_layers)]
        )

        self.activation = F.gelu
        self.normalization = make_normalization()
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x, basis_state = self.tokenizer(x_num, x_cat)

        for layer_idx, (layer, bp_gate) in enumerate(zip(self.layers, self.bp_gates)):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_stage1_residual = x
            h1 = layer["token_norm"](x)
            h1 = layer["linear0"](h1)
            h1 = self.activation(h1)
            h1 = layer["sgu"](h1)
            if self.ffn_dropout:
                h1 = F.dropout(h1, self.ffn_dropout, self.training)
            h1 = layer["linear1"](h1)
            if self.residual_dropout:
                h1 = F.dropout(h1, self.residual_dropout, self.training)
            x1 = x_stage1_residual + h1

            x_stage2_residual = x1
            h2 = layer["channel_norm"](x1)
            delta_c = layer["channel_mix"](h2)
            if self.residual_dropout:
                delta_c = F.dropout(delta_c, self.residual_dropout, self.training)

            if not isinstance(bp_gate, nn.Identity) and basis_state is not None:
                d_num = basis_state.shape[1]
                if d_num > 0:
                    bp_scale = bp_gate(basis_state)
                    delta_num = delta_c[:, 1 : 1 + d_num, :] * bp_scale
                    delta_c = delta_c.clone()
                    delta_c[:, 1 : 1 + d_num, :] = delta_num

            x = x_stage2_residual + self.channel_scales[layer_idx] * delta_c

        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x.squeeze(-1)


class GGPLTMLPTokChanBPGate(GGPLTMLP):
    """TokChan backbone with breakpoint-gated numeric channel residuals."""

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
                "ggpl_tmlp_tokchan_bpgate keeps a clean tokenizer replacement and does not support sparse gating options"
            )
        TabModel.__init__(self)
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTMLPTokChanBPGate(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tmlp_tokchan_bpgate"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get(
            "breakpoint_fallback", "quantile"
        )
        self.breakpoint_cache_dir = self.saved_model_config.get(
            "breakpoint_cache_dir", "artifacts/ggpl_tmlp_tokchan_bpgate_breakpoints"
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
        model_config.setdefault("d_channel", None)
        model_config.setdefault("channel_scale_init", 1e-2)
        model_config.setdefault("bp_gate_ratio", 0.25)
        model_config.setdefault("bp_gate_dropout", 0.0)
        model_config.setdefault("bp_gate_alpha_init", 1e-3)
        model_config.setdefault("bp_gate_zero_init", True)
        model_config.setdefault("bp_gate_all_layers", False)
        model_config.setdefault("bp_gate_start_layer", None)
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
