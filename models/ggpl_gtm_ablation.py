"""Legacy import compatibility for centralized HingeMix ablations."""

from .hingemix_ablation import (
    ABLATIONS,
    DynamicOnlyAblationBlock,
    HingeMixAblation,
    SharedLinearNumericTokenizer,
    _HingeMixAblation,
    resolve_ablation,
)

GGPLGTMAblation = HingeMixAblation
_GGPLGTMAblation = _HingeMixAblation

__all__ = [
    "ABLATIONS",
    "DynamicOnlyAblationBlock",
    "GGPLGTMAblation",
    "HingeMixAblation",
    "SharedLinearNumericTokenizer",
    "_GGPLGTMAblation",
    "_HingeMixAblation",
    "resolve_ablation",
]
