"""Fixed-name entry points for the centralized GGPL-GTM ablation implementations."""

import typing as ty

import torch

from .ggpl_gtm_ablation import GGPLGTMAblation


class _FixedGGPLGTMAblation(GGPLGTMAblation):
    """Bind one valid centralized ablation without duplicating model code."""

    model_name: str
    fixed_ablation: str

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
        config = dict(model_config)
        configured = config.get("ablation")
        if configured is not None and configured != self.fixed_ablation:
            raise ValueError(
                f"{self.model_name} fixes ablation={self.fixed_ablation!r}; "
                f"received incompatible ablation={configured!r}."
            )
        config["ablation"] = self.fixed_ablation
        super().__init__(
            model_config=config,
            n_num_features=n_num_features,
            categories=categories,
            n_labels=n_labels,
            device=device,
            feat_gate=feat_gate,
            pruning=pruning,
            dataset=dataset,
        )
        self.base_name = self.model_name
        self.saved_model_config["model_name"] = self.model_name


class GGPLGTMFullAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_full"
    fixed_ablation = "full"


class GGPLGTMNoChannelAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_no_channel"
    fixed_ablation = "no_channel"


class GGPLGTMNoGraphAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_no_graph"
    fixed_ablation = "no_graph"


class GGPLGTMNoGraphNoChannelAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_no_graph_no_channel"
    fixed_ablation = "no_graph_no_channel"


class GGPLGTMLinearAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_linear"
    fixed_ablation = "linear"


class GGPLGTMLinearNoChannelAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_linear_no_channel"
    fixed_ablation = "linear_no_channel"


class GGPLGTMLinearNoGraphAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_linear_no_graph"
    fixed_ablation = "linear_no_graph"


class GGPLGTMLinearNoGraphNoChannelAblation(_FixedGGPLGTMAblation):
    model_name = "ggpl_gtm_ablation_linear_no_graph_no_channel"
    fixed_ablation = "linear_no_graph_no_channel"
