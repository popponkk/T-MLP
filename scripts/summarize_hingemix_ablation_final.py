"""Summarize only the four-method formal HingeMix tau=16 ablation experiment."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DISPLAY = {"hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed", "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM", "stream": "STREAM", "cachesweep": "CacheSweep"}
LABELS = {"full": "HingeMix", "no_graph": "HingeMix-no_graph", "linear": "HingeMix-linear", "linear_no_graph": "HingeMix-linear_no_graph"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/ggpl_gtm_ablation_final_tau16")
    p.add_argument("--output", default="results/compare/hingemix_ablation_final_tau16")
    return p.parse_args()


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def mean_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def record(root, task):
    directory = root / task.get("source_experiment", task["experiment"]) / task["dataset"] / f"seed_{task['seed']}"
    marker = read(directory / "parameter_result.json")
    attempt = Path(marker["attempt_dir"]) if marker and isinstance(marker.get("attempt_dir"), str) else None
    meta, prediction, history = (read(attempt / "task.json") if attempt else None, read(attempt / "prediction.json") if attempt else None, read(attempt / "results.json") if attempt else None)
    status, reason = "missing", "result has not completed"
    if marker and marker.get("config_fingerprint") != task["config_fingerprint"]:
        status, reason = "conflict", "result fingerprint mismatch"
    elif meta and meta.get("config_fingerprint") == task["config_fingerprint"]:
        if isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
            status, reason = "completed", None
        else:
            status, reason = "invalid", "test RMSE missing or invalid"
    spec = task["spec"]
    effective = (meta or {}).get("effective_parameters", {})
    val = ((history or {}).get("val") or {})
    return {"model": "hingemix" if task["experiment"] == "full" else "hingemix_ablation", "source_model": spec["model"], "experiment": task["experiment"], "display_model": LABELS[task["experiment"]],
            "ablation": spec["ablation"], "tokenizer_type": spec["tokenizer"], "graph": spec["graph"], "channel": spec["channel"],
            "dataset": task["dataset"], "display_dataset": DISPLAY[task["dataset"]], "seed": task["seed"],
            "K": effective.get("K"), "r": effective.get("r"), "tau": effective.get("tau"), "d": effective.get("d"), "L": effective.get("L"),
            "validation_rmse": marker.get("validation_rmse") if marker else val.get("best_metric"), "test_rmse": marker.get("test_rmse") if marker else (prediction or {}).get("metrics", {}).get("rmse"),
            "best_epoch": marker.get("best_epoch") if marker else val.get("best_epoch"), "parameter_count": marker.get("parameter_count") if marker else None,
            "trainable_parameter_count": marker.get("trainable_parameter_count") if marker else None,
            "gpu": marker.get("gpu") if marker else task.get("gpu"), "train_time_seconds": marker.get("training_wall_seconds") if marker else (task.get("finished_at", 0) - task.get("started_at", 0) if task.get("finished_at") and task.get("started_at") else None),
            "status": status, "reason": reason, "attempt_dir": str(attempt) if attempt else None}


def main():
    args = parse_args(); root, output = Path(args.root), Path(args.output)
    manifest = read(root / "manifest.json")
    if not manifest:
        raise SystemExit("No ablation manifest found")
    legacy_map = {
        "full": "full",
        "no_graph": "no_graph",
        "shared_linear": "linear",
        "shared_linear_no_graph": "linear_no_graph",
    }
    source_tasks = manifest.get("tasks", [])
    if len(source_tasks) == 315:
        tasks = []
        for source in source_tasks:
            formal_name = legacy_map.get(source["experiment"])
            if formal_name is not None:
                task = dict(source)
                task["source_experiment"] = source["experiment"]
                task["experiment"] = formal_name
                tasks.append(task)
    elif len(source_tasks) == 140:
        tasks = source_tasks
    else:
        raise SystemExit("Expected a 315-task legacy or 140-task HingeMix tau=16 manifest")
    runs = [record(root, task) for task in tasks]
    fields = list(runs[0]); write(output / "per_seed.csv", runs, fields)
    missing = [row for row in runs if row["status"] != "completed"]; write(output / "missing_runs.csv", missing, fields)
    grouped = {}
    for row in runs: grouped.setdefault((row["experiment"], row["dataset"]), []).append(row)
    summaries = []
    for _, rows in sorted(grouped.items()):
        done = [row for row in rows if row["status"] == "completed"]
        example = rows[0]; val_mean, val_std = mean_std([r["validation_rmse"] for r in done]); test_mean, test_std = mean_std([r["test_rmse"] for r in done]); time_mean, time_std = mean_std([r["train_time_seconds"] for r in done])
        summaries.append({**{k: example[k] for k in ("model", "experiment", "display_model", "ablation", "tokenizer_type", "graph", "channel", "dataset", "display_dataset", "K", "r", "tau", "d", "L", "parameter_count", "trainable_parameter_count")},
                          "expected_seeds": 5, "successful_seeds": len(done), "missing_seeds": 5 - len(done), "complete": len(done) == 5,
                          "validation_rmse_mean": val_mean, "validation_rmse_std": val_std, "test_rmse_mean": test_mean, "test_rmse_std": test_std, "train_time_seconds_mean": time_mean, "train_time_seconds_std": time_std})
    write(output / "summary.csv", summaries)
    table = []
    for exp in LABELS:
        row = {"model": LABELS[exp]}
        for dataset, display in DISPLAY.items():
            match = next((x for x in summaries if x["experiment"] == exp and x["dataset"] == dataset), None)
            row[display] = None if match is None or match["test_rmse_mean"] is None else f"{match['test_rmse_mean']:.6g} +/- {match['test_rmse_std']:.3g}" if match["test_rmse_std"] is not None else f"{match['test_rmse_mean']:.6g}"
        table.append(row)
    write(output / "rmse_table.csv", table)
    integrity = {"expected_runs": 140, "completed_runs": len(runs) - len(missing), "missing_or_invalid_runs": len(missing), "expected_summaries": 28, "complete_five_seed_summaries": sum(x["complete"] for x in summaries), "source_manifest_tasks": len(source_tasks)}
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} runs, {len(summaries)} summaries, and {len(missing)} missing/invalid rows to {output}")


if __name__ == "__main__":
    main()
