from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolLinearTokenizerNoGraphNoChannel(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_linear_tokenizer_no_graph_no_channel"
    use_ggpl_tokenizer = False
    use_graph = False
    use_channel = False
