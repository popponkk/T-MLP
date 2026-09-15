from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationLinearTokenizerNoGraph(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_linear_tokenizer_no_graph"
    use_ggpl_tokenizer = False
    use_graph = False
    use_channel = True
