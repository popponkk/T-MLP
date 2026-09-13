from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolNoGraph(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_no_graph"
    use_ggpl_tokenizer = True
    use_graph = False
    use_channel = True
