from .ggpl_dynonly_ablation_shared_linear import _BaseGGPLDynOnlySharedLinearAblation


class GGPLDynOnlyAblationSharedLinearNoGraph(_BaseGGPLDynOnlySharedLinearAblation):
    model_name = "ggpl_dynonly_ablation_shared_linear_no_graph"
    use_graph = False
    use_channel = True
