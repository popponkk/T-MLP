from .ggpl_dynonly_ablation_shared_linear import _BaseGGPLDynOnlySharedLinearAblation


class GGPLDynOnlyAblationSharedLinearNoGraphNoChannel(
    _BaseGGPLDynOnlySharedLinearAblation
):
    model_name = "ggpl_dynonly_ablation_shared_linear_no_graph_no_channel"
    use_graph = False
    use_channel = False
