from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolLinearTokenizer(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_linear_tokenizer"
    use_ggpl_tokenizer = False
    use_graph = True
    use_channel = True
