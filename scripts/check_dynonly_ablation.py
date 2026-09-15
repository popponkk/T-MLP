"""CPU smoke checks for the CLS-readout dynamic-only factorial ablations."""

import torch

from models.ggpl_dynonly_ablation import (
    DynamicOnlyAblationBlock,
    _GGPLDynOnlyAblationModel,
)
from models.ggpl_tmlp_graph_slimtok_graph_ablation import GraphSlimTokAblationBlock
from models.ggpl_tmlp_graph_slimtok_graph_ablation import (
    _GGPLTMLPGraphSlimTokGraphAblation,
)
from utils.model_utils import MODEL_CARDS, make_baseline


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
REGISTRY_NAMES = {
    "ggpl_dynonly_ablation": (True, True, True),
    "ggpl_dynonly_ablation_no_channel": (True, True, False),
    "ggpl_dynonly_ablation_linear_tokenizer": (False, True, True),
    "ggpl_dynonly_ablation_no_graph": (True, False, True),
    "ggpl_dynonly_ablation_linear_tokenizer_no_channel": (False, True, False),
    "ggpl_dynonly_ablation_no_graph_no_channel": (True, False, False),
    "ggpl_dynonly_ablation_linear_tokenizer_no_graph": (False, False, True),
    "ggpl_dynonly_ablation_linear_tokenizer_no_graph_no_channel": (False, False, False),
}


def build(flags):
    tokenizer, graph, channel = flags
    return _GGPLDynOnlyAblationModel(
        d_numerical=4, categories=None, token_bias=True,
        use_ggpl_tokenizer=tokenizer, use_graph=graph, use_channel=channel,
        n_layers=1, d_token=8, graph_dynamic_rank=4, d_out=1,
    )


def copy_common_block_state(old, new):
    for name in (
        "graph_norm", "graph_dynamic_proj", "graph_out", "graph_scale",
        "channel_norm", "channel_down", "channel_up", "channel_scale",
    ):
        source = getattr(old, name)
        target = getattr(new, name)
        if isinstance(source, torch.nn.Parameter):
            target.data.copy_(source.data)
        else:
            target.load_state_dict(source.state_dict())


def main():
    torch.manual_seed(0)
    for name in REGISTRY_NAMES:
        assert name in MODEL_CARDS, name
        wrapped = make_baseline(
            name, {"d_token": 8, "n_layers": 1, "graph_dynamic_rank": 4},
            n_num=4, cat_card=None, n_labels=1, device="cpu",
        )
        assert wrapped.base_name == name, name
    for name, flags in VARIANTS.items():
        model = build(flags)
        x = torch.randn(3, 4, requires_grad=True)
        y = model(x)
        assert y.shape == (3,), (name, y.shape)
        y.square().mean().backward()
        block = model.layers[0]
        assert hasattr(block, "graph_norm") == flags[1], name
        assert hasattr(block, "channel_norm") == flags[2], name
        if not flags[0]:
            assert not hasattr(model.tokenizer, "set_breakpoints"), name
        # Capture the head input to prove that the readout is CLS, not pooling.
        captured = []
        handle = model.normalization.register_forward_pre_hook(
            lambda _, args: captured.append(args[0].detach())
        )
        readout_input = torch.randn(2, 4)
        model(readout_input)
        handle.remove()
        tokens = model.tokenizer(readout_input, None)
        for layer in model.layers:
            tokens = layer(tokens)
        assert torch.allclose(captured[0], tokens[:, 0]), name

        if not flags[1]:
            model.eval()
            x_a = torch.zeros(2, 4, requires_grad=True)
            x_b = torch.ones(2, 4, requires_grad=True)
            assert torch.allclose(model(x_a), model(x_b)), name
            model(x_a).sum().backward()
            assert x_a.grad is None or torch.allclose(x_a.grad, torch.zeros_like(x_a)), name

    # Align public dynamic-only weights to verify the retained graph/channel path.
    old = GraphSlimTokAblationBlock(
        n_tokens=5, d_token=8, channel_hidden=16, dropout=0.0,
        layerscale_init=1e-2, graph_dynamic_rank=4, graph_ablation="dynamic_only",
    )
    new = DynamicOnlyAblationBlock(
        n_tokens=5, d_token=8, channel_hidden=16, dropout=0.0, layerscale_init=1e-2,
        activation="gelu", graph_dynamic_rank=4, graph_temperature=1.0,
        graph_self_loop_init=2.0,
        use_graph=True, use_channel=True,
    )
    copy_common_block_state(old, new)
    tokens = torch.randn(2, 5, 8)
    old.eval()
    new.eval()
    assert torch.allclose(old(tokens), new(tokens), atol=1e-6)

    old_model = _GGPLTMLPGraphSlimTokGraphAblation(
        d_numerical=4, categories=None, token_bias=True, d_token=8, n_layers=1,
        graph_dynamic_rank=4, d_out=1, graph_ablation="dynamic_only",
    )
    new_model = build((True, True, True))
    new_model.load_state_dict(old_model.state_dict(), strict=True)
    x = torch.randn(3, 4)
    old_model.eval()
    new_model.eval()
    assert torch.allclose(old_model(x, None), new_model(x, None), atol=1e-6)
    print("All CLS-readout dynamic-only ablation smoke checks passed.")


if __name__ == "__main__":
    main()
