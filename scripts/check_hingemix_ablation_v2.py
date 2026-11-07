"""Lightweight CPU checks for the isolated HingeMix ablation v2 implementation."""

import copy
import json
import tempfile
from pathlib import Path

import torch

from models.hingemix import HingeMix
from models.hingemix_ablation import HingeMixAblation
from models.hingemix_ablation_v2 import (
    GraphChannelBlock,
    HingeMixAblationV2,
    TokenMLPChannelBlock,
)
from models.hingemix_token_mlp import HingeMixTokenMLP


BASE = {
    "n_layers": 1, "d_token": 16, "token_bias": True, "residual_dropout": 0.1,
    "num_breakpoints": 3, "learnable_breakpoints": True, "breakpoint_init": "quantile",
    "breakpoint_fallback": "quantile", "slimtok_channel_ratio": 2.0,
    "slimtok_layerscale_init": 1e-2, "slimtok_activation": "gelu",
    "graph_dynamic_rank": 4, "graph_temperature": 16.0, "token_mlp_ratio": 2.0,
}


def make_v2(mode: str, features: int):
    config = dict(BASE, ablation=mode)
    return HingeMixAblationV2(config, features, None, 1, device="cpu")


def copy_matching(target, source):
    state = source.state_dict()
    target.load_state_dict(state, strict=True)


def assert_close(left, right, label: str):
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-5):
        raise AssertionError(f"{label} differs after state transfer")


def check_equivalence():
    torch.manual_seed(7)
    x = torch.randn(3, 5)
    formal = HingeMix(dict(BASE), 5, None, 1, device="cpu")
    v2 = make_v2("full", 5)
    copy_matching(v2.model, formal.model)
    formal.model.eval(); v2.model.eval()
    assert_close(formal.model(x, None), v2.model(x, None), "full HingeMix")

    token = HingeMixTokenMLP(dict(BASE), 5, None, 1, device="cpu")
    v2_token = make_v2("token_mlp", 5)
    copy_matching(v2_token.model, token.model)
    token.model.eval(); v2_token.model.eval()
    assert_close(token.model(x, None), v2_token.model(x, None), "TokenMLP")

    linear = HingeMixAblation(dict(BASE, ablation="linear"), 5, None, 1, device="cpu")
    v2_linear = make_v2("linear", 5)
    copy_matching(v2_linear.model, linear.model)
    linear.model.eval(); v2_linear.model.eval()
    assert_close(linear.model(x, None), v2_linear.model(x, None), "shared Linear graph")


def check_modes_and_gradients():
    for features in (3, 7):
        for mode in ("full", "token_mlp", "linear", "linear_token_mlp"):
            model = make_v2(mode, features)
            x = torch.randn(4, features, requires_grad=True)
            output = model.model(x, None)
            if output.shape != (4,):
                raise AssertionError(f"{mode}: unexpected output shape {tuple(output.shape)}")
            output.square().mean().backward()
            if not torch.isfinite(x.grad).all():
                raise AssertionError(f"{mode}: input gradients are non-finite")
            blocks = list(model.model.layers)
            if mode.endswith("token_mlp"):
                if not all(isinstance(block, TokenMLPChannelBlock) for block in blocks):
                    raise AssertionError(f"{mode}: missing TokenMLP block")
                if any(hasattr(block, "graph_out") for block in blocks):
                    raise AssertionError(f"{mode}: retains graph branch")
            elif not all(isinstance(block, GraphChannelBlock) for block in blocks):
                raise AssertionError(f"{mode}: missing graph block")
            if mode.startswith("linear") and model.use_ggpl_tokenizer:
                raise AssertionError(f"{mode}: incorrectly enters breakpoint path")
            if not mode.startswith("linear") and not model.use_ggpl_tokenizer:
                raise AssertionError(f"{mode}: incorrectly skips breakpoint path")
            if mode.startswith("linear"):
                linear_layers = [item for item in model.model.tokenizer.modules() if isinstance(item, torch.nn.Linear)]
                if len(linear_layers) != 1:
                    raise AssertionError(f"{mode}: shared tokenizer must own exactly one Linear")
            if mode.endswith("token_mlp"):
                names = set(model.model.state_dict())
                if not {"layers.0.token_fc1.weight", "layers.0.token_fc2.weight", "layers.0.token_scale"} <= names:
                    raise AssertionError(f"{mode}: TokenMLP parameters are not registered")


def check_cls_path_and_round_trip():
    torch.manual_seed(9)
    for mode in ("token_mlp", "linear_token_mlp"):
        model = make_v2(mode, 4).model.eval()
        x = torch.zeros(2, 4, requires_grad=True)
        changed = x.detach().clone(); changed[0, 1] = 2.0
        tokens_a = model.layers[0](model.tokenizer(x, None))[:, 0]
        tokens_b = model.layers[0](model.tokenizer(changed, None))[:, 0]
        if torch.allclose(tokens_a, tokens_b):
            raise AssertionError(f"{mode}: CLS does not respond to numeric tokens")
        model(x, None).sum().backward()
        if not torch.isfinite(x.grad).all() or x.grad.abs().sum() == 0:
            raise AssertionError(f"{mode}: no finite input-to-output gradient")
        state = copy.deepcopy(model.state_dict())
        restored = make_v2(mode, 4).model.eval()
        restored.load_state_dict(state, strict=True)
        assert_close(model(x.detach(), None), restored(x.detach(), None), f"{mode} save/load")


def check_breakpoint_cache():
    with tempfile.TemporaryDirectory() as temporary:
        cache = str(Path(temporary) / "cache")
        config = dict(BASE, ablation="full", breakpoint_cache_dir=cache)
        first = HingeMixAblationV2(config, 3, None, 1, device="cpu")
        x, y = torch.randn(12, 3), torch.randn(12)
        first._fit_or_load_breakpoints(x, y, str(Path(temporary) / "run"))
        files = list(Path(cache).rglob("breakpoints_*.json"))
        if len(files) != 1:
            raise AssertionError("GGPL breakpoint cache was not written")
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        second = HingeMixAblationV2(config, 3, None, 1, device="cpu")
        second._fit_or_load_breakpoints(x, y, str(Path(temporary) / "run"))
        assert_close(first.model.tokenizer.breakpoints(), second.model.tokenizer.breakpoints(), "breakpoint cache hit")
        second._fit_or_load_breakpoints(x, y + 1, str(Path(temporary) / "run"))
        changed = json.loads(files[0].read_text(encoding="utf-8"))
        if changed["identity"]["train_y_sha256"] == payload["identity"]["train_y_sha256"]:
            raise AssertionError("incompatible breakpoint cache was not replaced")
        linear = make_v2("linear", 3)
        if linear.breakpoint_cache_dir is not None or hasattr(linear.model.tokenizer, "breakpoint_delta_raw"):
            raise AssertionError("shared Linear mode must not create breakpoint state")


def main():
    check_equivalence()
    check_modes_and_gradients()
    check_cls_path_and_round_trip()
    check_breakpoint_cache()
    try:
        make_v2("no_graph", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy no_graph must fail rather than map to TokenMLP")
    print("HingeMix ablation v2 checks passed.")


if __name__ == "__main__":
    main()
