from .ggpl_dynonly_pool import _BaseGGPLDynOnlyPool


class GGPLDynOnlyPoolNoGraphNoChannel(_BaseGGPLDynOnlyPool):
    model_name = "ggpl_dynonly_pool_no_graph_no_channel"
    use_ggpl_tokenizer = True
    use_graph = False
    use_channel = False
