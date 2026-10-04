"""Small CPU checks for formal HingeMix and all centralized ablations."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Running this file directly sets ``sys.path[0]`` to ``scripts/``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hingemix import _HingeMix
from models.ggpl_gtm import GGPLGTM, _GGPLGTM
from models.hingemix_ablation import (
    ABLATIONS,
    SharedLinearNumericTokenizer,
    _HingeMixAblation,
    resolve_ablation,
)
from utils.model_utils import MODEL_CARDS, make_baseline


def make_model(ablation: str):
    _, spec = resolve_ablation(ablation)
    return _HingeMixAblation(
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
    assert "hingemix" in MODEL_CARDS
    assert "hingemix_ablation" in MODEL_CARDS
    assert "ggpl_gtm" in MODEL_CARDS  # compatibility alias, not a formal display entry
    assert _GGPLGTM is _HingeMix
    assert MODEL_CARDS["ggpl_gtm"] is GGPLGTM
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
        if spec["use_graph"]:
            assert block.graph_temperature == 16.0
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
        "hingemix_ablation", config, n_num=4, cat_card=None, n_labels=1, device="cpu"
    )
    assert wrapper.ablation == "linear"
    assert wrapper.base_name == "hingemix_ablation_shared_linear/linear"
    complete = _HingeMix(
        d_numerical=4, categories=None, token_bias=True, d_token=8,
        n_layers=1, graph_dynamic_rank=4, d_out=1,
    )
    assert complete.layers[0].graph_temperature == 16.0
    full = make_model("full")
    full.load_state_dict(complete.state_dict(), strict=True)
    complete.eval()
    full.eval()
    x = torch.randn(3, 4)
    assert torch.allclose(complete(x, None), full(x, None), atol=1e-6)
    print("Formal HingeMix checks passed.")


if __name__ == "__main__":
    main()
