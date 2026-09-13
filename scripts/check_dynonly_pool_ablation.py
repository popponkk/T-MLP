"""CPU smoke checks for the eight dynamic-only pool ablation variants."""

import torch

from models.ggpl_dynonly_pool import DynamicOnlyPoolBlock, _GGPLDynOnlyPoolModel
from models.ggpl_tmlp_graph_slimtok_graph_ablation import GraphSlimTokAblationBlock


VARIANTS = {
    "ggpl": (True, True, True),
    "ggpl_no_channel": (True, True, False),
    "linear": (False, True, True),
    "ggpl_no_graph": (True, False, True),
    "linear_no_channel": (False, True, False),
    "ggpl_no_graph_no_channel": (True, False, False),
    "linear_no_graph": (False, False, True),
    "linear_no_graph_no_channel": (False, False, False),
}


def build(flags):
    use_ggpl, use_graph, use_channel = flags
    return _GGPLDynOnlyPoolModel(
        d_numerical=4,
        categories=None,
        token_bias=True,
        use_ggpl_tokenizer=use_ggpl,
        use_graph=use_graph,
        use_channel=use_channel,
        d_token=8,
        n_layers=1,
        graph_dynamic_rank=4,
        d_out=1,
    )


def main():
    torch.manual_seed(0)
    for name, flags in VARIANTS.items():
        model = build(flags)
        x = torch.randn(3, 4, requires_grad=True)
        y = model(x, None)
        assert y.shape == (3,), (name, y.shape)
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), name
        if not flags[1]:
            model.eval()
            x_a = torch.zeros(2, 4, requires_grad=True)
            x_b = torch.ones(2, 4, requires_grad=True)
            assert not torch.allclose(model(x_a), model(x_b)), name
            model(x_a).sum().backward()
            assert x_a.grad is not None and x_a.grad.abs().sum() > 0, name
        block = model.layers[0]
        assert hasattr(block, "graph_norm") == flags[1], name
        assert hasattr(block, "channel_norm") == flags[2], name
        if not flags[0]:
            assert not hasattr(model.tokenizer, "set_breakpoints"), name

    # Match common weights with the existing dynamic_only block before readout.
    old = GraphSlimTokAblationBlock(
        n_tokens=5, d_token=8, channel_hidden=16, dropout=0.0,
        layerscale_init=1e-2, graph_dynamic_rank=4, graph_ablation="dynamic_only",
    )
    new = DynamicOnlyPoolBlock(
        n_tokens=5, d_token=8, channel_hidden=16, dropout=0.0,
        layerscale_init=1e-2, activation="gelu", graph_dynamic_rank=4,
        graph_temperature=1.0, graph_self_loop_init=2.0,
        use_graph=True, use_channel=True,
    )
    new.load_state_dict(old.state_dict(), strict=False)
    tokens = torch.randn(2, 5, 8)
    old.eval()
    new.eval()
    assert torch.allclose(old(tokens), new(tokens), atol=1e-6)
    print("All eight pool-ablation smoke checks passed.")


if __name__ == "__main__":
    main()
