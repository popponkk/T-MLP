#!/usr/bin/env python3
"""Summarize only the isolated HingeMix training-set-size experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DISPLAY = {
    "hpcg2": "HPCG",
    "hpgmg3": "HPGMG",
    "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix",
    "raiderstream": "RaiderSTREAM",
    "stream": "STREAM",
    "cachesweep": "CacheSweep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_sample_size")
    parser.add_argument("--output", default="results/hingemix_sample_size_summary")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    valid = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return (
        statistics.mean(valid) if valid else None,
        statistics.stdev(valid) if len(valid) > 1 else None,
    )


def task_root(root: Path, task: dict[str, Any]) -> Path:
    return root / task["dataset"] / f"fraction_{task['subset']['percentage']:03d}pct" / f"seed_{task['training_seed']}"


def record(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    directory = task_root(root, task)
    marker = read_json(directory / "parameter_result.json")
    attempt = Path(marker["attempt_dir"]) if marker and isinstance(marker.get("attempt_dir"), str) else None
    metadata = read_json(attempt / "task.json") if attempt else None
    prediction = read_json(attempt / "prediction.json") if attempt else None
    history = read_json(attempt / "results.json") if attempt else None
    status, reason = "missing", "result has not completed"
    if marker and marker.get("config_fingerprint") != task["config_fingerprint"]:
        status, reason = "conflict", "result fingerprint mismatch"
    elif metadata and metadata.get("config_fingerprint") != task["config_fingerprint"]:
        status, reason = "conflict", "attempt metadata fingerprint mismatch"
    elif metadata and isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
        status, reason = "completed", None
    elif metadata:
        status, reason = "invalid", "test RMSE is missing or invalid"
    value = marker or {}
    subset = task["subset"]
    validation = (history or {}).get("val") or {}
    return {
        "model": "hingemix",
        "display_model": "HingeMix",
        "dataset": task["dataset"],
        "display_dataset": DISPLAY[task["dataset"]],
        "fraction_percent": subset["percentage"],
        "training_samples": subset["selected_train_size"],
        "full_training_samples": subset["source_train_size"],
        "training_seed": task["training_seed"],
        "data_seed": task["data_seed"],
        "subset_seed": subset["subset_seed"],
        "subset_positions_sha256": subset["train_positions_sha256"],
        "K": 8,
        "r": 16,
        "tau": 16.0,
        "d": 1024,
        "L": 1,
        "validation_rmse": value.get("validation_rmse", validation.get("best_metric")),
        "test_rmse": value.get("test_rmse", (prediction or {}).get("metrics", {}).get("rmse")),
        "best_epoch": value.get("best_epoch", validation.get("best_epoch")),
        "parameter_count": value.get("parameter_count"),
        "trainable_parameter_count": value.get("trainable_parameter_count"),
        "training_wall_seconds": value.get("training_wall_seconds"),
        "status": status,
        "reason": reason,
        "attempt_dir": str(attempt) if attempt else "",
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["fraction_percent"]), []).append(row)
    summaries = []
    for _, group in sorted(groups.items()):
        completed = [row for row in group if row["status"] == "completed"]
        example = completed[0] if completed else group[0]
        validation_mean, validation_std = mean_std([row["validation_rmse"] for row in completed])
        test_mean, test_std = mean_std([row["test_rmse"] for row in completed])
        time_mean, time_std = mean_std([row["training_wall_seconds"] for row in completed])
        seeds = sorted(row["training_seed"] for row in completed)
        summaries.append({
            "model": "hingemix",
            "display_model": "HingeMix",
            "dataset": example["dataset"],
            "display_dataset": example["display_dataset"],
            "fraction_percent": example["fraction_percent"],
            "training_samples": example["training_samples"],
            "full_training_samples": example["full_training_samples"],
            "expected_seeds": 5,
            "successful_seeds": len(completed),
            "successful_seed_values": ";".join(map(str, seeds)),
            "missing_seeds": 5 - len(completed),
            "complete": len(completed) == 5,
            "validation_rmse_mean": validation_mean,
            "validation_rmse_std": validation_std,
            "test_rmse_mean": test_mean,
            "test_rmse_std": test_std,
            "training_wall_seconds_mean": time_mean,
            "training_wall_seconds_std": time_std,
            "parameter_count": example["parameter_count"],
            "trainable_parameter_count": example["trainable_parameter_count"],
            "K": 8,
            "r": 16,
            "tau": 16.0,
            "d": 1024,
            "L": 1,
        })
    return summaries


def main() -> None:
    args = parse_args()
    root, output = Path(args.root), Path(args.output)
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest.get("experiment") != "hingemix_sample_size":
        raise SystemExit("No compatible HingeMix sample-size manifest found")
    expected = 7 * 7 * 5
    if manifest.get("expected_tasks") != expected:
        raise SystemExit("Manifest does not contain the fixed 245-task sample-size design")
    rows = [record(root, task) for task in manifest["tasks"]]
    fields = list(rows[0])
    write_csv(output / "per_seed.csv", rows, fields)
    write_csv(output / "missing_runs.csv", [row for row in rows if row["status"] != "completed"], fields)
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    curve = [{
        "model": item["display_model"], "dataset": item["display_dataset"],
        "fraction_percent": item["fraction_percent"], "training_samples": item["training_samples"],
        "test_rmse_mean": item["test_rmse_mean"], "test_rmse_std": item["test_rmse_std"],
        "successful_seeds": item["successful_seeds"], "complete": item["complete"],
    } for item in summary]
    write_csv(output / "learning_curve.csv", curve, list(curve[0]))
    table = []
    for fraction in sorted({item["fraction_percent"] for item in summary}):
        row: dict[str, Any] = {"fraction_percent": fraction}
        for dataset in DISPLAY.values():
            item = next((value for value in summary if value["display_dataset"] == dataset and value["fraction_percent"] == fraction), None)
            if item is None or item["test_rmse_mean"] is None:
                row[dataset] = ""
            elif item["test_rmse_std"] is None:
                row[dataset] = f"{item['test_rmse_mean']:.6g}"
            else:
                row[dataset] = f"{item['test_rmse_mean']:.6g} +/- {item['test_rmse_std']:.3g}"
        table.append(row)
    write_csv(output / "rmse_table.csv", table, list(table[0]))
    integrity = {
        "expected_runs": expected,
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "missing_or_invalid_runs": sum(row["status"] != "completed" for row in rows),
        "expected_dataset_fraction_summaries": 49,
        "complete_five_seed_summaries": sum(item["complete"] for item in summary),
    }
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} runs, {len(summary)} summaries, and "
          f"{integrity['missing_or_invalid_runs']} missing/invalid rows to {output}")


if __name__ == "__main__":
    main()
