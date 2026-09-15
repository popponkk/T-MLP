from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationNoChannel(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_no_channel"
    use_ggpl_tokenizer = True
    use_graph = True
    use_channel = False
