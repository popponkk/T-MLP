from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolLinearTokenizerNoChannel(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_linear_tokenizer_no_channel"
    use_ggpl_tokenizer = False
    use_graph = True
    use_channel = False
