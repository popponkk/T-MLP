"""Legacy import compatibility for the formal HingeMix model.

The implementation moved to :mod:`models.hingemix`; aliases preserve old
Python imports and state-dict-compatible model construction.
"""

from .hingemix import (
    DynamicOnlyGraphTokenChannelBlock,
    HingeMix,
    _HingeMix,
    _resolve_activation,
)

GGPLGTM = HingeMix
_GGPLGTM = _HingeMix

__all__ = [
    "DynamicOnlyGraphTokenChannelBlock",
    "GGPLGTM",
    "HingeMix",
    "_GGPLGTM",
    "_HingeMix",
    "_resolve_activation",
]
