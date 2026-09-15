"""CPU smoke checks for shared-linear CLS-readout ablations."""

import torch

from models.ggpl_dynonly_ablation import _GGPLDynOnlyAblationModel
from models.ggpl_dynonly_ablation_shared_linear import (
    SharedLinearNumericTokenizer,
    _GGPLDynOnlySharedLinearAblationModel,
)
from utils.model_utils import MODEL_CARDS, make_baseline


VARIANTS = {
    "ggpl_dynonly_ablation_shared_linear": (True, True),
    "ggpl_dynonly_ablation_shared_linear_no_channel": (True, False),
    "ggpl_dynonly_ablation_shared_linear_no_graph": (False, True),
    "ggpl_dynonly_ablation_shared_linear_no_graph_no_channel": (False, False),
}


def build(use_graph: bool, use_channel: bool):
    return _GGPLDynOnlySharedLinearAblationModel(
        d_numerical=4, categories=None, token_bias=True,
        use_graph=use_graph, use_channel=use_channel, n_layers=1,
        d_token=8, graph_dynamic_rank=4, d_out=1,
    )


def main():
    torch.manual_seed(0)
    tokenizer = SharedLinearNumericTokenizer(d_numerical=4, d_token=8, bias=True)
    x = torch.randn(3, 4)
    tokens = tokenizer(x)
    assert tokens.shape == (3, 5, 8)
    assert torch.allclose(tokens[:, 1:], tokenizer.linear(x.unsqueeze(-1)))
    identical_features = torch.randn(3, 1).expand(-1, 4)
    identical_tokens = tokenizer(identical_features)[:, 1:]
    assert torch.allclose(identical_tokens[:, 0], identical_tokens[:, 1])
    assert sum(parameter.numel() for parameter in tokenizer.parameters()) == 3 * 8
    assert set(dict(tokenizer.named_parameters())) == {
        "cls_token", "linear.weight", "linear.bias"
    }
    no_bias = SharedLinearNumericTokenizer(d_numerical=4, d_token=8, bias=False)
    assert sum(parameter.numel() for parameter in no_bias.parameters()) == 2 * 8

    for name, (use_graph, use_channel) in VARIANTS.items():
        assert name in MODEL_CARDS, name
        wrapped = make_baseline(
            name, {"d_token": 8, "n_layers": 1, "graph_dynamic_rank": 4},
            n_num=4, cat_card=None, n_labels=1, device="cpu",
        )
        assert wrapped.base_name == name
        model = build(use_graph, use_channel)
        inputs = torch.randn(3, 4, requires_grad=True)
        output = model(inputs)
        assert output.shape == (3,)
        output.square().mean().backward()
        assert not hasattr(model.tokenizer, "set_breakpoints")
        assert hasattr(model.tokenizer, "linear")
        block = model.layers[0]
        assert hasattr(block, "graph_norm") == use_graph
        assert hasattr(block, "channel_norm") == use_channel

        model.eval()
        permuted = torch.randperm(inputs.shape[1])
        assert torch.allclose(model(inputs), model(inputs[:, permuted]), atol=1e-6)
        if not use_graph:
            x_a = torch.zeros(2, 4, requires_grad=True)
            x_b = torch.ones(2, 4, requires_grad=True)
            assert torch.allclose(model(x_a), model(x_b), atol=1e-6)
            model(x_a).sum().backward()
            assert x_a.grad is None or torch.allclose(x_a.grad, torch.zeros_like(x_a))

    independent = _GGPLDynOnlyAblationModel(
        d_numerical=4, categories=None, token_bias=True,
        use_ggpl_tokenizer=False, use_graph=True, use_channel=True,
        d_token=8, n_layers=1, graph_dynamic_rank=4, d_out=1,
    )
    assert independent.tokenizer.__class__.__name__ == "LinearNumericTokenizer"
    print("Shared-linear dynamic-only ablation smoke checks passed.")


if __name__ == "__main__":
    main()
