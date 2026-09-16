from .ggpl_dynonly_ablation_independent_nnlinear import (
    _BaseGGPLDynOnlyIndependentNNLinearAblation,
)


class GGPLDynOnlyAblationIndependentNNLinearNoChannel(
    _BaseGGPLDynOnlyIndependentNNLinearAblation
):
    model_name = "ggpl_dynonly_ablation_independent_nnlinear_no_channel"
    use_graph = True
    use_channel = False
