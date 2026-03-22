import typing as ty

import torch
import torch.nn as nn

from .tmlp_variant_blocks import TMLPBackbone, SRHead, RegressionVariantTrainer


class _SRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        **backbone_config,
    ):
        super().__init__()
        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.head = SRHead(self.backbone.d_token, d_out)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        hidden, _ = self.backbone.encode_hidden(x_num, x_cat)
        predictions, extras = self.head(hidden)
        predictions = predictions.squeeze(-1)
        if return_extras:
            return predictions, extras
        return predictions, {}


class SRTMLP(RegressionVariantTrainer):
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
            raise NotImplementedError('tmlp-sr does not support sparse gating options yet')
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _SRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = 'tmlp-sr'
        self.device = torch.device(device)
