from .ggpl_dynonly_ablation import _BaseGGPLDynOnlyAblation


class GGPLDynOnlyAblationNoGraph(_BaseGGPLDynOnlyAblation):
    model_name = "ggpl_dynonly_ablation_no_graph"
    use_ggpl_tokenizer = True
    use_graph = False
    use_channel = True
