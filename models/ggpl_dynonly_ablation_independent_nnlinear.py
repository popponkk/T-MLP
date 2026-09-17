"""Independent-``nn.Linear`` tokenizer variants of the CLS-readout ablation."""

import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor

from .abstract import TabModel
from .ggpl_dynonly_ablation import DynamicOnlyAblationBlock, _BaseGGPLDynOnlyAblation, _resolve_activation


class IndependentNNLinearNumericTokenizer(nn.Module):
    """[CLS] plus one independently parameterized ``nn.Linear`` per feature."""

    def __init__(self, d_numerical: int, d_token: int, bias: bool = True) -> None:
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.d_token = int(d_token)
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
            raise NotImplementedError("Only numerical inputs are supported.")
        if x_num is None:
            raise ValueError("x_num is required")
        if x_num.ndim != 2 or x_num.shape[1] != self.d_numerical:
            raise ValueError(
                f"Expected x_num with shape [B, {self.d_numerical}], got {tuple(x_num.shape)}"
            )

        numeric_tokens = torch.stack(
            [linear(x_num[:, j : j + 1]) for j, linear in enumerate(self.linears)],
            dim=1,
        )
        cls = self.cls_token.view(1, 1, -1).expand(x_num.shape[0], 1, -1)
        return torch.cat([cls, numeric_tokens], dim=1)


class _GGPLDynOnlyIndependentNNLinearAblationModel(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        use_graph: bool,
        use_channel: bool,
        n_layers: int = 1,
        d_token: int = 1024,
        d_ffn_factor: float = 0.66,
        ffn_dropout: ty.Optional[float] = None,
        residual_dropout: ty.Optional[float] = 0.1,
        slimtok_channel_ratio: float = 2.0,
        slimtok_dropout: ty.Optional[float] = None,
        slimtok_layerscale_init: float = 1e-2,
        slimtok_activation: str = "gelu",
        graph_dynamic_rank: int = 16,
        graph_temperature: float = 1.0,
        graph_self_loop_init: float = 2.0,
        d_out: int = 1,
        **_: ty.Any,
    ) -> None:
        super().__init__()
        if categories:
            raise NotImplementedError(
                "Independent-nn.Linear ablations support pure numerical datasets only."
            )
        self.use_graph = bool(use_graph)
        self.use_channel = bool(use_channel)
        self.tokenizer = IndependentNNLinearNumericTokenizer(
            d_numerical=d_numerical, d_token=d_token, bias=token_bias
        )
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
                "Independent-nn.Linear ablations support pure numerical inputs only."
            )
        x = self.tokenizer(x_num, None)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.normalization(x)
        x = self.activation(x)
        return self.head(x).squeeze(-1)


class _BaseGGPLDynOnlyIndependentNNLinearAblation(_BaseGGPLDynOnlyAblation):
    """Use the existing linear-tokenizer fit path without breakpoint setup."""

    use_ggpl_tokenizer = False

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
        self.model = _GGPLDynOnlyIndependentNNLinearAblationModel(
            d_numerical=n_num_features,
            categories=None,
            d_out=n_labels,
            use_graph=self.use_graph,
            use_channel=self.use_channel,
            **config,
        ).to(device)
        self.base_name = self.model_name
        self.device = torch.device(device)
        self.breakpoint_init = None
        self.breakpoint_fallback = None
        self.breakpoint_cache_dir = None
        self.num_breakpoints = 0
        n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            f"[{self.model_name}] tokenizer=independent_nnlinear "
            f"graph={self.use_graph} channel={self.use_channel} "
            f"readout=cls parameters={n_parameters}"
        )


class GGPLDynOnlyAblationIndependentNNLinear(
    _BaseGGPLDynOnlyIndependentNNLinearAblation
):
    model_name = "ggpl_dynonly_ablation_independent_nnlinear"
    use_graph = True
    use_channel = True
