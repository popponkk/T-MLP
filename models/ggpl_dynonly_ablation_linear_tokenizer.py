from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationLinearTokenizer(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_linear_tokenizer"
    use_ggpl_tokenizer = False
    use_graph = True
    use_channel = True
