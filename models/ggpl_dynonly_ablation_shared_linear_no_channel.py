from .ggpl_dynonly_ablation_shared_linear import _BaseGGPLDynOnlySharedLinearAblation


class GGPLDynOnlyAblationSharedLinearNoChannel(
    _BaseGGPLDynOnlySharedLinearAblation
):
    model_name = "ggpl_dynonly_ablation_shared_linear_no_channel"
    use_graph = True
    use_channel = False
