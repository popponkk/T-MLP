from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolLinearTokenizerNoGraph(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_linear_tokenizer_no_graph"
    use_ggpl_tokenizer = False
    use_graph = False
    use_channel = True
