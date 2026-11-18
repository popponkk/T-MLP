#!/usr/bin/env python3
"""Synthetic, no-training checks for HingeMix breakpoint comparison support."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.hingemix import HingeMix  # noqa: E402


def make_model(strategy: str | None, cache_dir: Path, cache_key: str = "synthetic") -> HingeMix:
    config = {
        "model_name": "hingemix", "n_layers": 1, "d_token": 8,
        "num_breakpoints": 3, "learnable_breakpoints": True,
        "graph_dynamic_rank": 2, "graph_temperature": 16.0,
        "breakpoint_init": "gbdt", "breakpoint_fallback": "quantile",
        "breakpoint_strategy": strategy, "breakpoint_cache_dir": str(cache_dir),
        "breakpoint_cache_key": cache_key,
    }
    return HingeMix(config, n_num_features=2, categories=None, n_labels=1, device="cpu")


def fit_breakpoints(model: HingeMix, x: np.ndarray, root: Path) -> np.ndarray:
    model._fit_or_load_breakpoints(torch.tensor(x), torch.tensor([0.0, 1.0, 2.0, 3.0]), str(root))
    return model.model.tokenizer.breakpoints().detach().cpu().numpy()


def main() -> None:
    x = np.array([[0.0, 4.0], [2.0, 4.0], [4.0, 4.0], [10.0, 4.0]], dtype="float32")
    expected_quantile = np.array([[1.5, 3.0, 5.5], [4.0, 4.0, 4.0]], dtype="float32")
    expected_uniform = np.array([[2.5, 5.0, 7.5], [4.0, 4.0, 4.0]], dtype="float32")
    assert np.allclose(HingeMix._comparison_quantile_breakpoints(x, 3), expected_quantile)
    assert np.allclose(HingeMix._comparison_uniform_breakpoints(x, 3), expected_uniform)
    # Historical GBDT fallback remains the 0.1..0.9 implementation, not the
    # comparison quantile definition.
    assert np.allclose(HingeMix._quantile_breakpoints(x, 3)[0], [0.6, 3.0, 8.2])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        torch.manual_seed(9)
        quantile = make_model("quantile", root / "quantile")
        quantile_hash = quantile._non_breakpoint_state_sha256()
        quantile._extract_gbdt_thresholds = lambda *_: (_ for _ in ()).throw(AssertionError("quantile called GBDT"))
        effective_q = fit_breakpoints(quantile, x, root / "fit")
        assert effective_q.shape == (2, 3) and np.isfinite(effective_q).all()
        assert quantile.breakpoint_diagnostics["gbdt"]["called"] is False
        assert quantile.breakpoint_diagnostics["requested_unique_breakpoint_counts"] == [3, 1]
        torch.manual_seed(9)
        uniform = make_model("uniform", root / "uniform")
        assert uniform._non_breakpoint_state_sha256() == quantile_hash
        uniform._extract_gbdt_thresholds = lambda *_: (_ for _ in ()).throw(AssertionError("uniform called GBDT"))
        effective_u = fit_breakpoints(uniform, x, root / "fit")
        assert effective_u.shape == (2, 3) and np.isfinite(effective_u).all()
        assert uniform.breakpoint_diagnostics["gbdt"]["called"] is False
        assert sum(parameter.numel() for parameter in quantile.model.parameters()) == sum(parameter.numel() for parameter in uniform.model.parameters())
        torch.manual_seed(9)
        conflicting = make_model("quantile", root / "quantile", cache_key="different")
        try:
            fit_breakpoints(conflicting, x, root / "fit")
        except RuntimeError as error:
            assert "Refusing breakpoint-cache reuse" in str(error)
        else:
            raise AssertionError("cache identity mismatch was accepted")
        torch.manual_seed(9)
        historical = make_model(None, root / "historical")
        assert historical.breakpoint_strategy is None
        assert historical.breakpoint_init == "gbdt"
    print("HingeMix breakpoint ablation synthetic checks passed.")


if __name__ == "__main__":
    main()
