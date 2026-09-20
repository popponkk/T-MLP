"""Summarize complete, failed, and missing GGPL-GTM parameter-sensitivity runs."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


DISPLAY_DATASETS = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}
PARAMETER_CURVES = {
    "num_breakpoints": ("K", "num_breakpoints", 8),
    "graph_dynamic_rank": ("r", "graph_dynamic_rank", 16),
    "graph_temperature": ("tau", "graph_temperature", 1.0),
    "d_token": ("d", "d_token", 1024),
    "n_layers": ("L", "n_layers", 1),
}
PARAMETER_RE = re.compile(r"\[ggpl_gtm\] parameters=(\d+) trainable_parameters=(\d+)")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/ggpl_gtm_parameter")
    p.add_argument("--output", default="results/compare/ggpl_gtm_parameter")
    return p.parse_args()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def mean_std(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return (
        statistics.mean(values) if values else None,
        statistics.stdev(values) if len(values) >= 2 else None,
    )


def task_record(task):
    output = Path(task["output_dir"])
    prediction = read_json(output / "prediction.json")
    history = read_json(output / "results.json")
    metadata = read_json(output / "parameter_task.json")
    status = task.get("status", "missing")
    reason = task.get("failure_reason")
    rmse = None
    if status == "completed":
        if metadata is None or metadata.get("config_fingerprint") != task.get("config_fingerprint"):
            status, reason = "conflict", "missing or mismatched parameter_task.json"
        elif prediction is None or not isinstance((prediction.get("metrics") or {}).get("rmse"), (int, float)):
            status, reason = "invalid", "missing test RMSE in prediction.json"
        else:
            rmse = prediction["metrics"]["rmse"]
    validation = None
    compute_time = None
    if history:
        validation = ((history.get("val") or {}).get("best_metric"))
        compute_time = ((history.get("train") or {}).get("tot_time"))
    parameter_count = trainable_parameter_count = None
    log = output / "training.log"
    if log.exists():
        match = PARAMETER_RE.search(log.read_text(encoding="utf-8", errors="replace"))
        if match:
            parameter_count, trainable_parameter_count = map(int, match.groups())
    p = task["parameters"]
    return {
        "model": task.get("model", "ggpl_gtm"), "config_id": task["config_id"],
        "dataset": task["dataset"], "display_dataset": DISPLAY_DATASETS.get(task["dataset"], task["dataset"]),
        "seed": task["training_seed"], "K": p["num_breakpoints"], "r": p["graph_dynamic_rank"],
        "tau": p["graph_temperature"], "d": p["d_token"], "L": p["n_layers"],
        "validation_rmse": validation, "test_rmse": rmse,
        "parameter_count": parameter_count, "trainable_parameter_count": trainable_parameter_count,
        "train_time_seconds": task.get("finished_at", 0) - task.get("started_at", 0)
        if task.get("finished_at") and task.get("started_at") else None,
        "training_compute_seconds": compute_time, "status": status, "reason": reason,
        "output_dir": task["output_dir"], "config_fingerprint": task["config_fingerprint"],
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    options = args()
    root, output = Path(options.root), Path(options.output)
    manifest = read_json(root / "manifest.json")
    if not manifest:
        raise SystemExit(f"Missing or invalid manifest: {root / 'manifest.json'}")
    runs = [task_record(task) for task in manifest.get("tasks", [])]
    fields = list(runs[0]) if runs else []
    write_csv(output / "runs.csv", runs, fields)
    missing = [row for row in runs if row["status"] != "completed"]
    write_csv(output / "missing_runs.csv", missing, fields)

    grouped = {}
    for row in runs:
        grouped.setdefault((row["config_id"], row["dataset"]), []).append(row)
    summaries = []
    for (config_id, dataset), rows in sorted(grouped.items()):
        complete = [row for row in rows if row["status"] == "completed"]
        validation_mean, validation_std = mean_std([row["validation_rmse"] for row in complete])
        test_mean, test_std = mean_std([row["test_rmse"] for row in complete])
        time_mean, time_std = mean_std([row["train_time_seconds"] for row in complete])
        example = rows[0]
        summaries.append({
            "model": "ggpl_gtm", "config_id": config_id, "dataset": dataset,
            "display_dataset": example["display_dataset"], "K": example["K"], "r": example["r"],
            "tau": example["tau"], "d": example["d"], "L": example["L"],
            "expected_seeds": 3, "successful_seeds": len(complete),
            "missing_seeds": 3 - len(complete), "complete": len(complete) == 3,
            "validation_rmse_mean": validation_mean, "validation_rmse_std": validation_std,
            "test_rmse_mean": test_mean, "test_rmse_std": test_std,
            "train_time_mean": time_mean, "train_time_std": time_std,
            "parameter_count": example["parameter_count"],
            "trainable_parameter_count": example["trainable_parameter_count"],
        })
    summary_fields = list(summaries[0]) if summaries else []
    write_csv(output / "summary.csv", summaries, summary_fields)

    curves = []
    for parameter, (label, key, baseline_value) in PARAMETER_CURVES.items():
        for row in summaries:
            value = row[{"num_breakpoints": "K", "graph_dynamic_rank": "r", "graph_temperature": "tau", "d_token": "d", "n_layers": "L"}[key]]
            changed = sum([
                row["K"] != 8, row["r"] != 16, row["tau"] != 1.0,
                row["d"] != 1024, row["L"] != 1,
            ])
            if row["config_id"] == "baseline" or (changed == 1 and value != baseline_value):
                curves.append({"parameter": parameter, "label": label, "value": value,
                               "dataset": row["dataset"], "display_dataset": row["display_dataset"],
                               "test_rmse_mean": row["test_rmse_mean"], "test_rmse_std": row["test_rmse_std"],
                               "complete": row["complete"], "successful_seeds": row["successful_seeds"],
                               "config_id": row["config_id"]})
    curve_fields = list(curves[0]) if curves else ["parameter", "label", "value", "dataset"]
    write_csv(output / "parameter_curves.csv", curves, curve_fields)
    print(f"Wrote {len(runs)} runs, {len(summaries)} summaries, and {len(missing)} missing/invalid rows to {output}")


if __name__ == "__main__":
    main()
