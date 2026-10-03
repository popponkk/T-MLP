"""Lightweight structural checks for the independently loaded HingeMix-DP."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ggpl_tmlp import GGPLTokenizer as ReferenceGGPLTokenizer
from utils.model_utils import MODEL_CARDS, make_baseline


def main():
    torch.manual_seed(0)
    assert "hingemix_dp" in MODEL_CARDS
    wrapper = make_baseline(
        "hingemix_dp",
        {
            "d_token": 8,
            "n_layers": 1,
            "num_breakpoints": 3,
            "graph_dynamic_rank": 4,
            "graph_temperature": 16.0,
            "slimtok_channel_ratio": 2.0,
        },
        n_num=4,
        cat_card=None,
        n_labels=1,
        device="cpu",
    )
    model = wrapper.model
    block = model.layers[0]
    assert block.graph_temperature == 16.0
    assert block.head_proj is not block.tail_proj
    assert block.head_proj.weight.data_ptr() != block.tail_proj.weight.data_ptr()
    assert block.head_proj.weight.shape == (4, 8)
    assert block.tail_proj.weight.shape == (4, 8)
    assert not hasattr(block, "graph_logits")

    reference = ReferenceGGPLTokenizer(4, None, 8, True, 3, True)
    model.tokenizer.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(3, 4, requires_grad=True)
    assert torch.allclose(model.tokenizer(x, None), reference(x, None))
    prediction = model(x, None)
    assert prediction.shape == (3,)
    prediction.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.head_proj.weight.grad is not None
    assert block.tail_proj.weight.grad is not None
    print("HingeMix-DP checks passed.")


if __name__ == "__main__":
    main()
