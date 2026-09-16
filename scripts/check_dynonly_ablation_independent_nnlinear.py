"""CPU smoke checks for independent-``nn.Linear`` CLS-readout ablations."""

import torch
import torch.nn as nn

from models.ggpl_dynonly_ablation import DynamicOnlyAblationBlock
from models.ggpl_dynonly_ablation_independent_nnlinear import (
    IndependentNNLinearNumericTokenizer,
    _GGPLDynOnlyIndependentNNLinearAblationModel,
)
from utils.model_utils import MODEL_CARDS, make_baseline


VARIANTS = {
    "ggpl_dynonly_ablation_independent_nnlinear": (True, True),
    "ggpl_dynonly_ablation_independent_nnlinear_no_channel": (True, False),
    "ggpl_dynonly_ablation_independent_nnlinear_no_graph": (False, True),
    "ggpl_dynonly_ablation_independent_nnlinear_no_graph_no_channel": (False, False),
}


def build(use_graph: bool, use_channel: bool):
    return _GGPLDynOnlyIndependentNNLinearAblationModel(
        d_numerical=4,
        categories=None,
        token_bias=True,
        use_graph=use_graph,
        use_channel=use_channel,
        n_layers=1,
        d_token=8,
        graph_dynamic_rank=4,
        d_out=1,
    )


def assert_tokenizer_contract() -> None:
    d_numerical, d_token = 4, 8
    tokenizer = IndependentNNLinearNumericTokenizer(d_numerical, d_token, bias=True)
    assert isinstance(tokenizer.linears, nn.ModuleList)
    assert len(tokenizer.linears) == d_numerical
    assert all(
        isinstance(layer, nn.Linear)
        and layer.in_features == 1
        and layer.out_features == d_token
        for layer in tokenizer.linears
    )
    assert len({id(layer) for layer in tokenizer.linears}) == d_numerical
    assert len({layer.weight.data_ptr() for layer in tokenizer.linears}) == d_numerical
    assert len({layer.bias.data_ptr() for layer in tokenizer.linears}) == d_numerical
    assert sum(parameter.numel() for parameter in tokenizer.parameters()) == (
        2 * d_numerical + 1
    ) * d_token

    x = torch.randn(3, d_numerical)
    tokens = tokenizer(x)
    expected = torch.stack(
        [layer(x[:, j : j + 1]) for j, layer in enumerate(tokenizer.linears)], dim=1
    )
    assert tokens.shape == (3, 1 + d_numerical, d_token)
    assert torch.allclose(tokens[:, 1:], expected)
    assert torch.allclose(tokens[:, 0], tokenizer.cls_token.expand_as(tokens[:, 0]))

    baseline = tokenizer(x).detach().clone()
    with torch.no_grad():
        tokenizer.linears[2].weight.add_(1.0)
        tokenizer.linears[2].bias.add_(1.0)
    updated = tokenizer(x)
    assert torch.allclose(baseline[:, 1:3], updated[:, 1:3])
    assert not torch.allclose(baseline[:, 3], updated[:, 3])
    assert torch.allclose(baseline[:, 4:], updated[:, 4:])

    x_changed = x.clone()
    x_changed[:, 1] += 1.0
    before, after = tokenizer(x), tokenizer(x_changed)
    assert torch.allclose(before[:, :2], after[:, :2])
    assert not torch.allclose(before[:, 2], after[:, 2])
    assert torch.allclose(before[:, 3:], after[:, 3:])

    no_bias = IndependentNNLinearNumericTokenizer(d_numerical, d_token, bias=False)
    assert all(layer.bias is None for layer in no_bias.linears)
    assert sum(parameter.numel() for parameter in no_bias.parameters()) == (
        d_numerical + 1
    ) * d_token


def main() -> None:
    torch.manual_seed(0)
    assert_tokenizer_contract()

    for name, (use_graph, use_channel) in VARIANTS.items():
        assert name in MODEL_CARDS, name
        wrapped = make_baseline(
            name,
            {"d_token": 8, "n_layers": 1, "graph_dynamic_rank": 4},
            n_num=4,
            cat_card=None,
            n_labels=1,
            device="cpu",
        )
        assert wrapped.base_name == name
        model = build(use_graph, use_channel)
        assert isinstance(model.tokenizer, IndependentNNLinearNumericTokenizer)
        assert not hasattr(model.tokenizer, "set_breakpoints")
        assert not any("pool" in module_name.lower() for module_name, _ in model.named_modules())
        block = model.layers[0]
        assert isinstance(block, DynamicOnlyAblationBlock)
        assert hasattr(block, "graph_norm") == use_graph
        assert hasattr(block, "channel_norm") == use_channel
        assert not hasattr(block, "graph_logits") or (use_graph and use_channel)

        inputs = torch.randn(3, 4, requires_grad=True)
        output = model(inputs)
        assert output.shape == (3,)
        output.square().mean().backward()

        # No graph means CLS never receives a numerical-token information path.
        if not use_graph:
            model.eval()
            x_a = torch.zeros(2, 4, requires_grad=True)
            x_b = torch.ones(2, 4)
            assert torch.allclose(model(x_a), model(x_b), atol=1e-6)
            model(x_a).sum().backward()
            assert x_a.grad is None or torch.allclose(x_a.grad, torch.zeros_like(x_a))

    for existing in (
        "ggpl_dynonly_ablation",
        "ggpl_dynonly_ablation_linear_tokenizer",
        "ggpl_dynonly_ablation_shared_linear",
    ):
        assert existing in MODEL_CARDS
    print("Independent nn.Linear dynamic-only ablation smoke checks passed.")


if __name__ == "__main__":
    main()
