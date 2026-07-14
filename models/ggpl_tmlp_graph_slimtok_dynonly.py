from .ggpl_tmlp_graph_slimtok_graph_ablation import (
    GGPLTMLPGraphSlimTokDynamicOnly,
)


class GGPLTMLPGraphSlimTokDynOnly(GGPLTMLPGraphSlimTokDynamicOnly):
    """Thin wrapper that formalizes the existing dynamic_only path."""

    model_name = "ggpl_tmlp_graph_slimtok_dynonly"
    breakpoint_dirname = "ggpl_tmlp_graph_slimtok_dynonly_breakpoints"
