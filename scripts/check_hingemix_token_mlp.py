"""CPU-only structural checks for the Hinge-Mix-TokenMLP experiment variant."""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hingemix_token_mlp import (  # noqa: E402
    HingeMixTokenMLP,
    TokenMLPChannelBlock,
    _HingeMixTokenMLP,
)
from utils.model_utils import MODEL_CARDS, make_baseline  # noqa: E402


def make_model(d_numerical: int, categories=None) -> _HingeMixTokenMLP:
    return _HingeMixTokenMLP(
        d_numerical=d_numerical,
        categories=categories,
        token_bias=True,
        n_layers=1,
        d_token=8,
        num_breakpoints=3,
        token_mlp_ratio=2.0,
        slimtok_channel_ratio=2.0,
        slimtok_layerscale_init=1e-2,
        slimtok_activation="gelu",
        residual_dropout=0.1,
        d_out=1,
    )


def check_shapes_gradients_and_state() -> None:
    for d_numerical, categories in ((2, None), (5, [3, 2])):
        model = make_model(d_numerical, categories)
        model.train()
        x_num = torch.randn(3, d_numerical, requires_grad=True)
        x_cat = None
        if categories is not None:
            x_cat = torch.stack(
                [torch.randint(cardinality, (3,)) for cardinality in categories], dim=1
            )
        prediction = model(x_num, x_cat)
        assert prediction.shape == (3,)
        prediction.square().mean().backward()
        assert x_num.grad is not None and torch.isfinite(x_num.grad).all()

        block = model.layers[0]
        assert block.n_tokens == model.tokenizer.n_tokens
        assert block.hidden_tokens == max(1, int(block.n_tokens * 2.0))
        keys = set(model.state_dict())
        assert any(key.endswith("token_fc1.weight") for key in keys)
        assert any(key.endswith("token_fc2.weight") for key in keys)
        assert any(key.endswith("token_scale") for key in keys)
        assert not any(
            forbidden in key
            for key in keys
            for forbidden in ("graph_dynamic_proj", "graph_out", "graph_scale", "graph_logits")
        )
        optimizer_ids = {id(parameter) for group in torch.optim.AdamW(model.parameters()).param_groups for parameter in group["params"]}
        assert id(block.token_fc1.weight) in optimizer_ids
        assert id(block.token_fc2.weight) in optimizer_ids
        assert id(block.token_scale) in optimizer_ids

        model.eval()
        probe_num = torch.randn(2, d_numerical)
        probe_cat = x_cat[:2] if x_cat is not None else None
        expected = model(probe_num, probe_cat)
        clone = make_model(d_numerical, categories).eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        actual = clone(probe_num, probe_cat)
        assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-6)


def check_cls_token_path() -> None:
    # Controlled weights force token 1 to contribute to the CLS output token.
    block = TokenMLPChannelBlock(
        d_token=4,
        n_tokens=3,
        token_mlp_ratio=2.0,
        channel_hidden=8,
        dropout=0.0,
        layerscale_init=1.0,
        activation="gelu",
    ).eval()
    with torch.no_grad():
        block.token_fc1.weight.zero_()
        block.token_fc1.bias.zero_()
        block.token_fc1.weight[0, 1] = 1.0
        block.token_fc2.weight.zero_()
        block.token_fc2.bias.zero_()
        block.token_fc2.weight[0, 0] = 1.0
        block.channel_scale.zero_()

    x = torch.tensor(
        [
            [[0.3, -0.2, 0.7, 1.1], [1.0, 2.0, 3.0, 5.0], [2.0, -1.0, 0.0, 4.0]],
            [[0.3, -0.2, 0.7, 1.1], [5.0, 3.0, 2.0, 1.0], [2.0, -1.0, 0.0, 4.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = block(x)
    assert not torch.allclose(output[0, 0], output[1, 0])
    output[:, 0, 0].sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad[:, 1, :].abs() > 0)


def check_registry_config_and_breakpoints() -> None:
    config_path = REPO_ROOT / "configs" / "default" / "hingemix_token_mlp.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model"]["model_name"] == "hingemix_token_mlp"
    assert config["model"]["token_mlp_ratio"] == 2.0
    assert "graph_dynamic_rank" not in config["model"]
    assert "graph_temperature" not in config["model"]
    assert MODEL_CARDS["hingemix_token_mlp"] is HingeMixTokenMLP

    wrapper = make_baseline(
        "hingemix_token_mlp",
        config["model"],
        n_num=3,
        cat_card=None,
        n_labels=1,
        device="cpu",
    )
    assert wrapper.base_name == "hingemix_token_mlp"
    with tempfile.TemporaryDirectory() as temporary:
        wrapper.breakpoint_cache_dir = temporary
        wrapper._fit_or_load_breakpoints(torch.randn(12, 3), torch.randn(12), temporary)
        assert list(Path(temporary).rglob("breakpoints_f3_b8.json"))


def main() -> None:
    torch.manual_seed(0)
    check_shapes_gradients_and_state()
    check_cls_token_path()
    check_registry_config_and_breakpoints()
    print("HingeMix TokenMLP checks passed.")


if __name__ == "__main__":
    main()
