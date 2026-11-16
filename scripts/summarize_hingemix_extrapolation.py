#!/usr/bin/env python3
"""Summarize verified HingeMix configuration-extrapolation tasks and draw RMSE curves."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_extrapolation_common import read_json, write_csv, write_json  # noqa: E402

LABELS = {"hingemix": "HingeMix", "lightgbm": "LightGBM", "mlp": "MLP"}
COLORS = {"hingemix": "#0b6e69", "lightgbm": "#cc5b2e", "mlp": "#345995"}
MARKERS = {"hingemix": "o", "lightgbm": "s", "mlp": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", default="results/hingemix_extrapolation")
    parser.add_argument("--output-root", default="results/hingemix_extrapolation_summary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def main() -> None:
    args = parse_args()
    plan = __import__("yaml").safe_load(Path(args.plan).read_text(encoding="utf-8"))
    experiment = str(plan.get("experiment_id") or f"{plan['source']['dataset']}_{plan['extrapolation']['parameter']}_extrapolation")
    results_root = Path(args.results_root).resolve()
    task_manifest = read_json(results_root / experiment / "tasks.json")
    if not isinstance(task_manifest, dict):
        raise SystemExit("No task manifest; run_hingemix_extrapolation.py must prepare/run tasks first")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_root).resolve() / experiment / timestamp
    per_seed, missing, per_config = [], [], []
    expected_seeds = set(plan.get("training_seeds") or [0, 1, 2, 3, 4])
    for task in task_manifest.get("tasks", []):
        root = Path(task.get("attempt_dir", ""))
        if task.get("status") != "completed":
            missing.append({"model": task.get("model"), "seed": task.get("seed"), "reason": task.get("reason", task.get("status")), "task_path": str(root)})
            continue
        prediction = read_json(root / "prediction.json")
        if not isinstance(prediction, dict):
            missing.append({"model": task["model"], "seed": task["seed"], "reason": "prediction missing", "task_path": str(root)})
            continue
        metrics = prediction.get("metrics") or {}
        try:
            rmse, mae = float(metrics["rmse"]), float(metrics["mae"])
        except (KeyError, TypeError, ValueError):
            missing.append({"model": task["model"], "seed": task["seed"], "reason": "invalid final metrics", "task_path": str(root)})
            continue
        per_seed.append({"model": task["model"], "display_model": LABELS.get(task["model"], task["model"]), "dataset": plan["source"]["dataset"], "parameter": plan["extrapolation"]["parameter"], "seed": task["seed"], "test_rmse": rmse, "test_mae": mae, "manifest_fingerprint": task["manifest_fingerprint"], "attempt_path": str(root)})
        path = root / "per_test_configuration.csv"
        if not path.is_file():
            missing.append({"model": task["model"], "seed": task["seed"], "reason": "per-test configuration metrics missing", "task_path": str(root)})
            continue
        for row in read_csv(path):
            per_config.append({"model": task["model"], "display_model": LABELS.get(task["model"], task["model"]), "dataset": plan["source"]["dataset"], "parameter": plan["extrapolation"]["parameter"], "seed": task["seed"], **row})
    write_csv(output / "per_seed.csv", per_seed, ["model", "display_model", "dataset", "parameter", "seed", "test_rmse", "test_mae", "manifest_fingerprint", "attempt_path"])
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_model[row["model"]].append(row)
    summaries = []
    for model in task_manifest.get("tasks", []):
        name = model["model"]
        if any(row["model"] == name for row in summaries):
            continue
        rows = by_model.get(name, [])
        seed_set = {row["seed"] for row in rows}
        complete = seed_set == expected_seeds and len(rows) == len(expected_seeds)
        rmse_mean, rmse_std = mean_std([row["test_rmse"] for row in rows])
        mae_mean, mae_std = mean_std([row["test_mae"] for row in rows])
        summaries.append({"model": name, "display_model": LABELS.get(name, name), "dataset": plan["source"]["dataset"], "parameter": plan["extrapolation"]["parameter"], "n_valid_seeds": len(rows), "expected_seeds": ",".join(map(str, sorted(expected_seeds))), "test_rmse_mean": rmse_mean if complete else "", "test_rmse_std_ddof1": rmse_std if complete else "", "test_mae_mean": mae_mean if complete else "", "test_mae_std_ddof1": mae_std if complete else "", "status": "complete" if complete else "incomplete"})
    write_csv(output / "summary.csv", summaries, ["model", "display_model", "dataset", "parameter", "n_valid_seeds", "expected_seeds", "test_rmse_mean", "test_rmse_std_ddof1", "test_mae_mean", "test_mae_std_ddof1", "status"])
    curves = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_config:
        grouped[(row["model"], row["configuration_value"])].append(row)
    for (model, value), rows in grouped.items():
        seeds = {int(row["seed"]) for row in rows}
        complete = seeds == expected_seeds
        rmse_values, mae_values = [float(row["rmse"]) for row in rows], [float(row["mae"]) for row in rows]
        rmse_mean, rmse_std = mean_std(rmse_values)
        mae_mean, mae_std = mean_std(mae_values)
        curves.append({"model": model, "display_model": LABELS.get(model, model), "dataset": plan["source"]["dataset"], "parameter": plan["extrapolation"]["parameter"], "configuration_value": value, "n_valid_seeds": len(seeds), "rmse_mean": rmse_mean if complete else "", "rmse_std_ddof1": rmse_std if complete else "", "mae_mean": mae_mean if complete else "", "mae_std_ddof1": mae_std if complete else "", "status": "complete" if complete else "incomplete"})
    curves.sort(key=lambda row: (row["model"], float(row["configuration_value"])))
    write_csv(output / "per_test_configuration.csv", per_config, ["model", "display_model", "dataset", "parameter", "seed", "configuration_value", "n_samples", "rmse", "mae"])
    write_csv(output / "configuration_rmse_curve.csv", curves, ["model", "display_model", "dataset", "parameter", "configuration_value", "n_valid_seeds", "rmse_mean", "rmse_std_ddof1", "mae_mean", "mae_std_ddof1", "status"])
    write_csv(output / "missing_runs.csv", missing, ["model", "seed", "reason", "task_path"])
    plot_status = {"created": False, "reason": "not attempted"}
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 4))
        for model in sorted({row["model"] for row in curves}):
            rows = [row for row in curves if row["model"] == model and row["status"] == "complete"]
            if len(rows) < 2:
                continue
            axis.plot([float(row["configuration_value"]) for row in rows], [float(row["rmse_mean"]) for row in rows], label=LABELS.get(model, model), color=COLORS.get(model), marker=MARKERS.get(model, "o"))
        axis.set_xlabel(plan["extrapolation"]["parameter"])
        axis.set_ylabel("Test RMSE")
        axis.set_title(f"{plan['source']['dataset']}: configuration extrapolation")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout(); figure.savefig(output / "configuration_rmse_curve.png", dpi=180); plt.close(figure)
        plot_status = {"created": True, "path": str(output / "configuration_rmse_curve.png")}
    except ImportError:
        plot_status = {"created": False, "reason": "matplotlib unavailable; CSV curve was written"}
    write_json(output / "integrity.json", {"expected_models": sorted({task["model"] for task in task_manifest.get("tasks", [])}), "expected_seeds": sorted(expected_seeds), "per_seed_rows": len(per_seed), "missing_or_invalid": len(missing), "plot": plot_status})
    print(f"Wrote extrapolation summary to {output}")


if __name__ == "__main__":
    main()
