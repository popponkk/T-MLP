#!/usr/bin/env python3
"""Read new no-IC data versions and run a small HingeMix forward pass only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.processor import DataProcessor  # noqa: E402
from utils.model_utils import load_config_from_file, make_baseline  # noqa: E402


DATASETS = ("raiderstream_no_ic", "cachesweep_no_ic")


def main() -> None:
    for name in DATASETS:
        directory = ROOT / "data" / "datasets" / name
        if not directory.is_dir():
            directory = ROOT / "data" / "custom_datasets" / name
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        removed = {column.strip().lower() for column in provenance["removed_columns"]}
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        names = info.get("num_feature_names", info.get("numerical_feature_names", []))
        if removed & {column.strip().lower() for column in names}:
            raise RuntimeError(f"{name}: removed fields remain in info.json")
        if len(names) != provenance["kept_feature_count"]:
            raise RuntimeError(f"{name}: feature count conflicts with provenance")
        output = ROOT / "results" / "hingemix_no_ic" / "preflight" / name
        dataset = DataProcessor.load_preproc_default(output, "hingemix", name, seed=42)
        config = load_config_from_file(ROOT / "configs" / "default" / "hingemix.yaml")
        config["model"]["breakpoint_cache_dir"] = str(output / "breakpoint_cache")
        model = make_baseline(
            "hingemix", config["model"], dataset.n_num_features, None, 1,
            dataset=dataset, device=torch.device("cpu"),
        )
        x_num, _, _, _ = DataProcessor.prepare(dataset, model)["train"]
        with torch.no_grad():
            result = model.model(x_num[:2], None)
        if result.shape[0] != min(2, x_num.shape[0]):
            raise RuntimeError(f"{name}: invalid forward output shape {tuple(result.shape)}")
        print(f"validated {name}: n_num={dataset.n_num_features} forward={tuple(result.shape)} cache={output}")


if __name__ == "__main__":
    main()
