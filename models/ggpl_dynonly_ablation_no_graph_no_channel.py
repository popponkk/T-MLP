from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationNoGraphNoChannel(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_no_graph_no_channel"
    use_ggpl_tokenizer = True
    use_graph = False
    use_channel = False
