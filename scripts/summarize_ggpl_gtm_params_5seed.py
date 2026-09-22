"""Merge the original 399 GGPL-GTM runs with five-seed supplement results."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DISPLAY = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}
BASELINE = {"K": 8, "r": 16, "tau": 1.0, "d": 1024, "L": 1}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root", default="results/ggpl_gtm_parameter")
    p.add_argument("--supplement-root", default="results/ggpl_gtm_parameter_supplement")
    p.add_argument("--output", default="results/compare/ggpl_gtm_parameter_5seed")
    return p


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def mean_std(values):
    values = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def output_for(task, supplemental):
    root = Path(task["output_dir"]) if not supplemental else None
    if supplemental:
        root = Path(task["_task_root"])
        marker = read_json(root / "parameter_result.json")
        if marker and isinstance(marker.get("attempt_dir"), str):
            return root, Path(marker["attempt_dir"]), marker
        return root, None, marker
    return root, root, read_json(root / "parameter_result.json")


def record(task, supplemental=False):
    root, attempt, marker = output_for(task, supplemental)
    status, reason, prediction = "missing", "result has not been completed", None
    if marker and marker.get("config_fingerprint") == task.get("config_fingerprint") and attempt:
        meta, prediction = read_json(attempt / "parameter_task.json"), read_json(attempt / "prediction.json")
        if meta and meta.get("config_fingerprint") == task.get("config_fingerprint"):
            rmse = (prediction or {}).get("metrics", {}).get("rmse")
            if isinstance(rmse, (int, float)):
                status, reason = "completed", None
            else:
                status, reason = "invalid", "test RMSE missing or invalid"
        else:
            status, reason = "conflict", "task metadata fingerprint mismatch"
    elif marker:
        status, reason = "conflict", "result fingerprint mismatch"
    params = task["parameters"]
    history = read_json(attempt / "results.json") if attempt else None
    validation = ((history or {}).get("val") or {}).get("best_metric")
    best_epoch = ((history or {}).get("val") or {}).get("best_epoch")
    compute_time = ((history or {}).get("train") or {}).get("tot_time")
    return {
        "model": "ggpl_gtm", "source": "supplement" if supplemental else "legacy",
        "group": task.get("group", "legacy_seed012"), "config_id": task["config_id"],
        "dataset": task["dataset"], "display_dataset": DISPLAY.get(task["dataset"], task["dataset"]),
        "seed": task["training_seed"], "K": params["num_breakpoints"], "r": params["graph_dynamic_rank"],
        "tau": params["graph_temperature"], "d": params["d_token"], "L": params["n_layers"],
        "validation_rmse": validation,
        "test_rmse": (prediction or {}).get("metrics", {}).get("rmse"),
        "best_epoch": best_epoch,
        "train_time_seconds": task.get("finished_at", 0) - task.get("started_at", 0)
        if task.get("finished_at") and task.get("started_at") else None,
        "training_compute_seconds": compute_time,
        "status": status,
        "reason": reason, "output_dir": str(root), "attempt_dir": str(attempt) if attempt else None,
        "config_fingerprint": task["config_fingerprint"],
    }


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    options = parser().parse_args()
    legacy_root, supplement_root, output = map(Path, (options.legacy_root, options.supplement_root, options.output))
    legacy = read_json(legacy_root / "manifest.json")
    supplement = read_json(supplement_root / "manifest.json")
    if not legacy or len(legacy.get("tasks", [])) != 399:
        raise SystemExit("Legacy manifest must contain exactly 399 tasks")
    if not supplement or len(supplement.get("tasks", [])) != 476:
        raise SystemExit("Supplement manifest must contain exactly 476 tasks")
    supplement_tasks = []
    for task in supplement["tasks"]:
        task = dict(task)
        task["_task_root"] = str(supplement_root / task["group"] / task["config_id"] / task["dataset"] / f"seed_{task['training_seed']}")
        supplement_tasks.append(task)
    legacy_rows = [record(task) for task in legacy["tasks"]]
    supplement_rows = [record(task, supplemental=True) for task in supplement_tasks]
    all_rows = legacy_rows + supplement_rows
    run_fields = list(all_rows[0]) if all_rows else []
    write_csv(output / "supplement_runs.csv", supplement_rows, run_fields)
    write_csv(output / "all_runs.csv", all_rows, run_fields)
    missing = [row for row in all_rows if row["status"] != "completed"]
    write_csv(output / "missing_runs.csv", missing, run_fields)

    groups = {}
    for row in all_rows:
        groups.setdefault((row["config_id"], row["dataset"]), []).append(row)
    summaries = []
    for (config_id, dataset), rows in sorted(groups.items()):
        successful = [row for row in rows if row["status"] == "completed"]
        values = [row["test_rmse"] for row in successful]
        avg, std = mean_std(values)
        validation_avg, validation_std = mean_std([row["validation_rmse"] for row in successful])
        time_avg, time_std = mean_std([row["train_time_seconds"] for row in successful])
        example = rows[0]
        seeds = sorted(row["seed"] for row in successful)
        summaries.append({
            "model": "ggpl_gtm", "config_id": config_id, "dataset": dataset,
            "display_dataset": example["display_dataset"], "K": example["K"], "r": example["r"],
            "tau": example["tau"], "d": example["d"], "L": example["L"],
            "expected_seeds": 5, "successful_seeds": len(successful),
            "missing_seeds": 5 - len(successful), "completed_seed_values": ";".join(map(str, seeds)),
            "complete": len(successful) == 5 and seeds == [0, 1, 2, 3, 4],
            "validation_rmse_mean": validation_avg, "validation_rmse_std": validation_std,
            "test_rmse_mean": avg, "test_rmse_std": std,
            "train_time_mean": time_avg, "train_time_std": time_std,
        })
    write_csv(output / "summary_5seed.csv", summaries)

    curves = []
    field_for = {"K": "K", "r": "r", "tau": "tau", "d": "d", "L": "L"}
    for row in summaries:
        changed = sum(row[key] != value for key, value in BASELINE.items())
        if row["config_id"] == "baseline" or changed == 1:
            for label, field in field_for.items():
                if row["config_id"] == "baseline" or row[field] != BASELINE[label]:
                    curves.append({"parameter": label, "value": row[field], **row})
    write_csv(output / "parameter_curves_5seed.csv", curves)
    atomic = {
        "expected_all_runs": 875, "actual_rows": len(all_rows), "completed_rows": len(all_rows) - len(missing),
        "missing_or_invalid_rows": len(missing), "configuration_dataset_summaries": len(summaries),
        "complete_five_seed_summaries": sum(row["complete"] for row in summaries),
    }
    (output / "integrity.json").write_text(json.dumps(atomic, indent=2), encoding="utf-8")
    print(f"Wrote {len(supplement_rows)} supplement rows, {len(all_rows)} combined rows, "
          f"{len(summaries)} summaries, and {len(missing)} missing/invalid rows to {output}")


if __name__ == "__main__":
    main()
