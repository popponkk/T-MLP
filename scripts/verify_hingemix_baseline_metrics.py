"""Verify and summarize direct test RMSE/MAE for formal HingeMix full runs.

This script is read-only with respect to the ablation results.  It follows the
task-root ``parameter_result.json`` marker selected by the v2 runner, then
checks the selected attempt's metadata and final test ``prediction.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover - documented runtime dependency
    raise SystemExit("PyYAML is required by this project to read final_config.yaml") from error


DATASETS = (
    "hpcg2",
    "hpgmg3",
    "ramspeed",
    "mix_with_five_datasets161",
    "raiderstream",
    "stream",
    "cachesweep",
)
DISPLAY = {
    "hpcg2": "HPCG",
    "hpgmg3": "HPGMG",
    "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix",
    "raiderstream": "RaiderSTREAM",
    "stream": "STREAM",
    "cachesweep": "CacheSweep",
}
EXPECTED_RMSE = {
    "hpcg2": 0.02002101665333647,
    "hpgmg3": 0.013854357457543393,
    "ramspeed": 0.011485329495479863,
    "mix_with_five_datasets161": 0.007665311307480479,
    "raiderstream": 0.0003309839408929696,
    "stream": 0.017080622898503605,
    "cachesweep": 0.020269089865870202,
}
SEEDS = {0, 1, 2, 3, 4}
EXPECTED_MODEL = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
}
EXPECTED_TRAINING = {"lr": 1e-5, "batch_size": 32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_ablation_v2")
    parser.add_argument("--output", default="results/hingemix_baseline_metrics_verified")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"cannot read {path}: {error}"
    return (value, None) if isinstance(value, dict) else (None, f"{path} is not a JSON object")


def read_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return None, f"cannot read {path}: {error}"
    return (value, None) if isinstance(value, dict) else (None, f"{path} is not a YAML mapping")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same_number(actual: Any, expected: float | int) -> bool:
    return finite_number(actual) and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def failure_row(task: dict[str, Any], reason: str, source: Path) -> dict[str, Any]:
    return {
        "dataset": task.get("dataset"),
        "display_name": DISPLAY.get(task.get("dataset"), task.get("dataset")),
        "seed": task.get("seed"),
        "configuration": task.get("ablation"),
        "test_rmse": None,
        "test_mae": None,
        "metric_scale": "unknown",
        "source_path": str(source.resolve()),
        "attempt_path": None,
        "extraction_method": "not_extracted",
        "verification_status": "invalid",
        "reason": reason,
    }


def verify_task(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    task_root = root / "full" / str(task.get("dataset")) / f"seed_{task.get('seed')}"
    marker_path = task_root / "parameter_result.json"
    if task.get("model") != "hingemix_ablation_v2" or task.get("ablation") != "full":
        return failure_row(task, "manifest task is not formal hingemix_ablation_v2/full", marker_path)
    if task.get("spec") != {"tokenizer_type": "ggpl", "mixing_type": "graph"}:
        return failure_row(task, "manifest full specification is not ggpl + graph", marker_path)
    if task.get("data_seed") != 42:
        return failure_row(task, "data_seed is not 42", marker_path)
    if any(not same_number((task.get("parameters") or {}).get(key), value) for key, value in EXPECTED_MODEL.items()):
        return failure_row(task, "manifest model parameters conflict with formal HingeMix", marker_path)
    if any(not same_number(task.get(key), value) for key, value in EXPECTED_TRAINING.items()):
        return failure_row(task, "manifest training parameters conflict with formal HingeMix", marker_path)

    marker, error = read_json(marker_path)
    if error:
        return failure_row(task, error, marker_path)
    if marker.get("config_fingerprint") != task.get("config_fingerprint"):
        return failure_row(task, "task-root result fingerprint mismatch", marker_path)
    attempt_value = marker.get("attempt_dir")
    if not isinstance(attempt_value, str):
        return failure_row(task, "selected attempt path is missing", marker_path)
    attempt = Path(attempt_value)
    metadata, error = read_json(attempt / "task.json")
    if error:
        return failure_row(task, error, marker_path)
    if metadata.get("config_fingerprint") != task.get("config_fingerprint"):
        return failure_row(task, "selected attempt fingerprint mismatch", marker_path)
    if metadata.get("ablation") != "full" or metadata.get("spec") != task["spec"]:
        return failure_row(task, "selected attempt is not the formal full configuration", marker_path)

    config, error = read_yaml(attempt / "final_config.yaml")
    if error:
        return failure_row(task, error, marker_path)
    model_config, training_config = config.get("model"), config.get("training")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        return failure_row(task, "selected attempt lacks model/training final config", marker_path)
    if any(not same_number(model_config.get(key), value) for key, value in EXPECTED_MODEL.items()):
        return failure_row(task, "selected attempt model config conflicts with formal HingeMix", marker_path)
    if any(not same_number(training_config.get(key), value) for key, value in EXPECTED_TRAINING.items()):
        return failure_row(task, "selected attempt training config conflicts with formal HingeMix", marker_path)

    prediction, error = read_json(attempt / "prediction.json")
    if error:
        return failure_row(task, error, marker_path)
    metrics = prediction.get("metrics")
    if prediction.get("metric_name") != "rmse" or not isinstance(metrics, dict):
        return failure_row(task, "prediction is not a final regression RMSE record", marker_path)
    rmse, mae = metrics.get("rmse"), metrics.get("mae")
    if not finite_number(rmse) or not finite_number(mae):
        return failure_row(task, "final test RMSE or MAE is missing/non-finite", marker_path)
    if not same_number(prediction.get("metric"), rmse):
        return failure_row(task, "prediction metric and metrics.rmse conflict", marker_path)
    if finite_number(marker.get("test_rmse")) and not same_number(marker["test_rmse"], rmse):
        return failure_row(task, "marker test_rmse and prediction metrics.rmse conflict", marker_path)
    return {
        "dataset": task["dataset"],
        "display_name": DISPLAY[task["dataset"]],
        "seed": task["seed"],
        "configuration": "full",
        "test_rmse": rmse,
        "test_mae": mae,
        "metric_scale": "original_target_scale (standardized error multiplied by training-subset y_std)",
        "source_path": str(marker_path.resolve()),
        "attempt_path": str(attempt.resolve()),
        "extraction_method": "direct: selected attempt prediction.json metrics.rmse + metrics.mae",
        "verification_status": "verified",
        "reason": "selected by task-root marker and matching manifest/attempt fingerprints",
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        valid_rmse = [float(row["test_rmse"]) for row in group if row["verification_status"] == "verified"]
        valid_mae = [float(row["test_mae"]) for row in group if row["verification_status"] == "verified"]
        valid_seeds = {int(row["seed"]) for row in group if row["verification_status"] == "verified"}
        rmse_mean = statistics.mean(valid_rmse) if len(valid_rmse) == 5 and valid_seeds == SEEDS else None
        mae_mean = statistics.mean(valid_mae) if len(valid_mae) == 5 and valid_seeds == SEEDS else None
        expected = EXPECTED_RMSE[dataset]
        difference = abs(rmse_mean - expected) if rmse_mean is not None else None
        if rmse_mean is None or mae_mean is None:
            status = "incomplete"
        elif difference is not None and difference > 1e-12:
            status = "rmse_mismatch"
        else:
            status = "verified"
        summaries.append({
            "dataset": dataset,
            "display_name": DISPLAY[dataset],
            "n_valid_rmse": len(valid_rmse),
            "n_valid_mae": len(valid_mae),
            "valid_seeds": ",".join(map(str, sorted(valid_seeds))),
            "rmse_mean": rmse_mean,
            "mae_mean": mae_mean,
            "expected_rmse_mean": expected,
            "rmse_abs_diff": difference,
            "status": status,
        })
    return summaries


def report(output: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]], root: Path) -> None:
    verified = sum(row["verification_status"] == "verified" for row in rows)
    lines = [
        "# HingeMix Formal Baseline Metrics Verification",
        "",
        f"- Result root: `{root.resolve()}`",
        "- Selected configuration: `hingemix_ablation_v2`, `full`, GGPL tokenizer, graph mixing, Channel Mixing, CLS readout.",
        "- Selection rule: task-root `parameter_result.json` -> its `attempt_dir`, with matching manifest and attempt fingerprints.",
        "- Metric source: selected attempt `prediction.json.metrics.rmse` and `.mae`.",
        "- Scale: original target scale. `utils/metrics.py` multiplies standardized RMSE/MAE by the training-split `y_std`.",
        "- Checkpoint rule: source code selects `best.bin` by validation RMSE; final test prediction is evaluated after loading it.",
        f"- Verified seed records: {verified}/35.",
        "",
        "## Per-dataset status",
        "",
        "| Dataset | Valid RMSE seeds | Valid MAE seeds | RMSE check | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for item in summaries:
        difference = "待核实" if item["rmse_abs_diff"] is None else f"{item['rmse_abs_diff']:.3g}"
        lines.append(f"| {item['display_name']} | {item['n_valid_rmse']} | {item['n_valid_mae']} | {difference} | {item['status']} |")
    lines += ["", "## Paper table", "", "| Dataset | RMSE | MAE |", "|---|---:|---:|"]
    for item in summaries:
        rmse = f"{item['rmse_mean']:.4f}" if item["status"] == "verified" else "待核实"
        mae = f"{item['mae_mean']:.4f}" if item["status"] == "verified" else "待核实"
        lines.append(f"| {item['display_name']} | {rmse} | {mae} |")
    lines += [
        "",
        "## Implementation evidence",
        "",
        "- `models/abstract.py:652-680`: validation metric selects the best checkpoint.",
        "- `main.py:267-278`: reloads the best checkpoint, performs final test evaluation, and writes `prediction.json`.",
        "- `utils/metrics.py:14-29,71-77`: RMSE and MAE are both restored using the same `y_std`.",
        "- `scripts/run_hingemix_ablation_v2.py:107-121`: valid completed result requires matching fingerprint and selected final test RMSE.",
    ]
    (output / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, output: Path, overwrite: bool) -> None:
    manifest, error = read_json(root / "manifest.json")
    if error:
        raise SystemExit(error)
    if manifest.get("experiment") != "hingemix_ablation_v2" or manifest.get("expected_tasks") != 140:
        raise SystemExit("No compatible 140-task hingemix_ablation_v2 manifest found")
    tasks = [task for task in manifest.get("tasks", []) if task.get("ablation") == "full"]
    if len(tasks) != 35:
        raise SystemExit(f"Expected exactly 35 full tasks, found {len(tasks)}")
    seen = {(task.get("dataset"), task.get("seed")) for task in tasks}
    if len(seen) != 35:
        raise SystemExit("Manifest full tasks have duplicate dataset/seed identities")
    rows = [verify_task(root, task) for task in sorted(tasks, key=lambda item: (DATASETS.index(item["dataset"]), item["seed"]))]
    summaries = summarize(rows)
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {output}; pass --overwrite-output to replace generated verification files")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    fields = [
        "dataset", "display_name", "seed", "configuration", "test_rmse", "test_mae", "metric_scale",
        "source_path", "attempt_path", "extraction_method", "verification_status", "reason",
    ]
    csv_write(output / "per_seed_metrics.csv", rows, fields)
    csv_write(output / "summary_metrics.csv", summaries, list(summaries[0]))
    report(output, rows, summaries, root)
    print(f"Wrote {len(rows)} per-seed records and {len(summaries)} summaries to {output}")
    print("statuses:", {status: sum(item["status"] == status for item in summaries) for status in sorted({item["status"] for item in summaries})})


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "results"
        task = {
            "model": "hingemix_ablation_v2", "ablation": "full", "spec": {"tokenizer_type": "ggpl", "mixing_type": "graph"},
            "dataset": "hpcg2", "seed": 0, "data_seed": 42, "parameters": {**EXPECTED_MODEL, "token_mlp_ratio": 2.0},
            "lr": 1e-5, "batch_size": 32, "config_fingerprint": "test",
        }
        task_root = root / "full" / "hpcg2" / "seed_0"
        attempt = task_root / "attempts" / "attempt_001"
        attempt.mkdir(parents=True)
        (task_root / "parameter_result.json").write_text(json.dumps({"config_fingerprint": "test", "attempt_dir": str(attempt), "test_rmse": 1.0}), encoding="utf-8")
        (attempt / "task.json").write_text(json.dumps(task), encoding="utf-8")
        (attempt / "final_config.yaml").write_text(yaml.safe_dump({"model": EXPECTED_MODEL, "training": EXPECTED_TRAINING}), encoding="utf-8")
        (attempt / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.0, "metrics": {"rmse": 1.0, "mae": 0.5}}), encoding="utf-8")
        row = verify_task(root, task)
        assert row["verification_status"] == "verified" and row["test_mae"] == 0.5
        (attempt / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.0, "metrics": {"rmse": 1.0}}), encoding="utf-8")
        assert verify_task(root, task)["verification_status"] == "invalid"
    print("Self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(Path(args.root), Path(args.output), args.overwrite_output)


if __name__ == "__main__":
    main()
