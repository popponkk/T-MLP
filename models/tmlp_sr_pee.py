import typing as ty

import torch
import torch.nn as nn

from .tmlp_variant_blocks import TMLPBackbone, PEEHead, RegressionVariantTrainer


class _SRPEETMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        pee_n_heads: int = 4,
        pee_learned_weight: bool = False,
        **backbone_config,
    ):
        super().__init__()
        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.head = PEEHead(
            self.backbone.d_token,
            d_out,
            n_heads=pee_n_heads,
            learned_weight=pee_learned_weight,
        )

    def forward(self, x_num, x_cat, return_extras: bool = False):
        hidden, _ = self.backbone.encode_hidden(x_num, x_cat)
        predictions, extras = self.head(hidden)
        predictions = predictions.squeeze(-1)
        if return_extras:
            return predictions, extras
        return predictions, {}


class SRPEETMLP(RegressionVariantTrainer):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = 'cuda',
        feat_gate: ty.Optional[str] = None,
        pruning: ty.Optional[str] = None,
        dataset=None,
    ):
        if feat_gate or pruning:
            raise NotImplementedError('tmlp-sr-pee does not support sparse gating options yet')
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _SRPEETMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = 'tmlp-sr-pee'
        self.device = torch.device(device)
