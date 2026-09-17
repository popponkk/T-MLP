from .ggpl_dynonly_ablation_independent_nnlinear import (
    _BaseGGPLDynOnlyIndependentNNLinearAblation,
)


class GGPLDynOnlyAblationIndependentNNLinearNoGraphNoChannel(
    _BaseGGPLDynOnlyIndependentNNLinearAblation
):
    model_name = "ggpl_dynonly_ablation_independent_nnlinear_no_graph_no_channel"
    use_graph = False
    use_channel = False
