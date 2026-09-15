from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationLinearTokenizerNoGraphNoChannel(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_linear_tokenizer_no_graph_no_channel"
    use_ggpl_tokenizer = False
    use_graph = False
    use_channel = False
