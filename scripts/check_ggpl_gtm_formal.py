"""Small CPU checks for formal GGPL-GTM and all centralized ablations."""

import torch
import torch.nn as nn

from models.ggpl_gtm import _GGPLGTM
from models.ggpl_gtm_ablation import (
    ABLATIONS,
    SharedLinearNumericTokenizer,
    _GGPLGTMAblation,
    resolve_ablation,
)
from utils.model_utils import MODEL_CARDS, make_baseline


DERIVED_VARIANTS = {
    "ggpl_gtm_ablation_full": "full",
    "ggpl_gtm_ablation_no_channel": "no_channel",
    "ggpl_gtm_ablation_no_graph": "no_graph",
    "ggpl_gtm_ablation_no_graph_no_channel": "no_graph_no_channel",
    "ggpl_gtm_ablation_linear": "linear",
    "ggpl_gtm_ablation_linear_no_channel": "linear_no_channel",
    "ggpl_gtm_ablation_linear_no_graph": "linear_no_graph",
    "ggpl_gtm_ablation_linear_no_graph_no_channel": "linear_no_graph_no_channel",
}


def make_model(ablation: str):
    _, spec = resolve_ablation(ablation)
    return _GGPLGTMAblation(
        d_numerical=4,
        categories=None,
        token_bias=True,
        tokenizer_type=spec["tokenizer_type"],
        use_graph=spec["use_graph"],
        use_channel=spec["use_channel"],
        n_layers=1,
        d_token=8,
        graph_dynamic_rank=4,
        d_out=1,
    )


def check_shared_linear() -> None:
    tokenizer = SharedLinearNumericTokenizer(4, 8, bias=True)
    assert isinstance(tokenizer.linear, nn.Linear)
    assert tokenizer.linear.in_features == 1
    assert tokenizer.linear.out_features == 8
    assert sum(parameter.numel() for parameter in tokenizer.linear.parameters()) == 2 * 8
    assert sum(parameter.numel() for parameter in tokenizer.parameters()) == 3 * 8
    assert sum(
        parameter.numel()
        for parameter in SharedLinearNumericTokenizer(4, 8, bias=False).parameters()
    ) == 2 * 8

    x = torch.tensor([[0.5, 0.5, -1.0, 2.0]])
    numeric_tokens = tokenizer(x)[:, 1:]
    assert torch.allclose(numeric_tokens[:, 0], numeric_tokens[:, 1])


def main() -> None:
    torch.manual_seed(0)
    assert set(ABLATIONS) == {
        "full", "no_channel", "no_graph", "no_graph_no_channel",
        "linear", "linear_no_channel", "linear_no_graph", "linear_no_graph_no_channel",
    }
    assert "ggpl_gtm" in MODEL_CARDS
    assert "ggpl_gtm_ablation" in MODEL_CARDS
    check_shared_linear()

    for name, spec in ABLATIONS.items():
        model = make_model(name)
        x = torch.randn(3, 4, requires_grad=True)
        y = model(x)
        assert y.shape == (3,)
        y.square().mean().backward()
        block = model.layers[0]
        assert hasattr(block, "graph_norm") == spec["use_graph"]
        assert hasattr(block, "channel_norm") == spec["use_channel"]
        assert not hasattr(block, "graph_logits")
        assert not any("pool" in module_name.lower() for module_name, _ in model.named_modules())
        if not spec["use_graph"]:
            model.eval()
            x_a = torch.zeros(2, 4, requires_grad=True)
            x_b = torch.ones(2, 4)
            assert torch.allclose(model(x_a), model(x_b), atol=1e-6)
            model(x_a).sum().backward()
            assert x_a.grad is None or torch.allclose(x_a.grad, torch.zeros_like(x_a))

    config = {"d_token": 8, "n_layers": 1, "graph_dynamic_rank": 4, "ablation": "linear"}
    wrapper = make_baseline(
        "ggpl_gtm_ablation", config, n_num=4, cat_card=None, n_labels=1, device="cpu"
    )
    assert wrapper.ablation == "linear"
    assert wrapper.base_name == "ggpl_gtm_ablation_shared_linear/linear"
    for model_name, ablation in DERIVED_VARIANTS.items():
        assert model_name in MODEL_CARDS
        derived = make_baseline(
            model_name,
            {"d_token": 8, "n_layers": 1, "graph_dynamic_rank": 4, "ablation": ablation},
            n_num=4,
            cat_card=None,
            n_labels=1,
            device="cpu",
        )
        assert derived.ablation == ablation
        assert derived.base_name == model_name

    complete = _GGPLGTM(
        d_numerical=4, categories=None, token_bias=True, d_token=8,
        n_layers=1, graph_dynamic_rank=4, d_out=1,
    )
    full = make_model("full")
    full.load_state_dict(complete.state_dict(), strict=True)
    complete.eval()
    full.eval()
    x = torch.randn(3, 4)
    assert torch.allclose(complete(x, None), full(x, None), atol=1e-6)
    print("Formal GGPL-GTM checks passed.")


if __name__ == "__main__":
    main()
