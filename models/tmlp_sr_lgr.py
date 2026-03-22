import typing as ty

import torch
import torch.nn as nn

from .tmlp_variant_blocks import TMLPBackbone, LatentGroupRefiner, SRHead, RegressionVariantTrainer


class _SRLGRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        lgr_n_groups: int = 8,
        lgr_dropout: float = 0.1,
        **backbone_config,
    ):
        super().__init__()
        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.refiner = LatentGroupRefiner(
            n_features=d_numerical + (0 if categories is None else len(categories)),
            d_token=self.backbone.d_token,
            n_groups=lgr_n_groups,
            dropout=lgr_dropout,
        )
        self.final_norm = nn.LayerNorm(self.backbone.d_token)
        self.final_act = nn.GELU()
        self.head = SRHead(self.backbone.d_token, d_out)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        token_states = self.backbone.encode_tokens(x_num, x_cat)
        refined_tokens, lgr_extras = self.refiner(token_states)
        hidden = refined_tokens[:, 0]
        hidden = self.final_norm(hidden)
        hidden = self.final_act(hidden)
        predictions, extras = self.head(hidden)
        extras.update(lgr_extras)
        predictions = predictions.squeeze(-1)
        if return_extras:
            return predictions, extras
        return predictions, {}


class SRLGRTMLP(RegressionVariantTrainer):
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
            raise NotImplementedError('tmlp-sr-lgr does not support sparse gating options yet')
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _SRLGRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = 'tmlp-sr-lgr'
        self.device = torch.device(device)
