from .ggpl_dynonly_ablation_independent_nnlinear import (
    _BaseGGPLDynOnlyIndependentNNLinearAblation,
)


class GGPLDynOnlyAblationIndependentNNLinearNoGraph(
    _BaseGGPLDynOnlyIndependentNNLinearAblation
):
    model_name = "ggpl_dynonly_ablation_independent_nnlinear_no_graph"
    use_graph = False
    use_channel = True
