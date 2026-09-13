from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolNoChannel(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_no_channel"
    use_ggpl_tokenizer = True
    use_graph = True
    use_channel = False
