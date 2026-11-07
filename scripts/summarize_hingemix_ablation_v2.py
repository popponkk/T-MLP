"""Summarize only the four isolated HingeMix ablation v2 configurations."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161", "raiderstream", "stream", "cachesweep")
DISPLAY = {"hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed", "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM", "stream": "STREAM", "cachesweep": "CacheSweep"}
LABELS = {"full": "HingeMix", "token_mlp": "HingeMix-token_mlp", "linear": "HingeMix-linear", "linear_token_mlp": "HingeMix-linear_token_mlp"}
SEEDS = {0, 1, 2, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_ablation_v2")
    parser.add_argument("--output", default="results/hingemix_ablation_v2_summary")
    return parser.parse_args()


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def run_record(root: Path, task: dict) -> dict:
    directory = root / task["ablation"] / task["dataset"] / f"seed_{task['seed']}"
    marker = read(directory / "parameter_result.json") or {}
    attempt = Path(marker["attempt_dir"]) if isinstance(marker.get("attempt_dir"), str) else None
    meta = read(attempt / "task.json") if attempt else None
    prediction = read(attempt / "prediction.json") if attempt else None
    history = read(attempt / "results.json") if attempt else None
    status, reason = "missing", "result has not completed"
    if marker and marker.get("config_fingerprint") != task["config_fingerprint"]:
        status, reason = "conflict", "result fingerprint mismatch"
    elif meta and meta.get("config_fingerprint") == task["config_fingerprint"]:
        test_rmse = (prediction or {}).get("metrics", {}).get("rmse")
        if isinstance(test_rmse, (int, float)) and math.isfinite(test_rmse):
            status, reason = "completed", None
        else:
            status, reason = "invalid", "test RMSE missing, non-finite, or invalid"
    effective = (meta or {}).get("effective_parameters", {})
    validation = (history or {}).get("val") or {}
    return {"model": "hingemix_ablation_v2", "ablation": task["ablation"], "display_model": LABELS[task["ablation"]],
            "tokenizer_type": task["spec"]["tokenizer_type"], "mixing_type": task["spec"]["mixing_type"],
            "channel": True, "readout": "cls", "dataset": task["dataset"], "display_dataset": DISPLAY[task["dataset"]],
            "seed": task["seed"], "K": effective.get("K"), "r": effective.get("r"), "tau": effective.get("tau"),
            "token_mlp_ratio": effective.get("token_mlp_ratio"), "d": effective.get("d"), "L": effective.get("L"),
            "validation_rmse": marker.get("validation_rmse", validation.get("best_metric")),
            "test_rmse": marker.get("test_rmse", (prediction or {}).get("metrics", {}).get("rmse")),
            "best_epoch": marker.get("best_epoch", validation.get("best_epoch")),
            "parameter_count": marker.get("parameter_count"), "trainable_parameter_count": marker.get("trainable_parameter_count"),
            "token_mlp_parameter_count": marker.get("token_mlp_parameter_count"),
            "channel_mlp_parameter_count": marker.get("channel_mlp_parameter_count"), "status": status, "reason": reason,
            "attempt_dir": str(attempt) if attempt else None}


def main():
    args = parse_args()
    root, output = Path(args.root), Path(args.output)
    manifest = read(root / "manifest.json")
    if not manifest or manifest.get("expected_tasks") != 140:
        raise SystemExit("No compatible HingeMix ablation v2 manifest found")
    rows = [run_record(root, task) for task in manifest["tasks"]]
    fields = list(rows[0])
    write(output / "per_seed.csv", rows, fields)
    write(output / "missing_runs.csv", [row for row in rows if row["status"] != "completed"], fields)
    summaries = []
    for ablation in LABELS:
        for dataset in DATASETS:
            group = [row for row in rows if row["ablation"] == ablation and row["dataset"] == dataset]
            done = [row for row in group if row["status"] == "completed"]
            found_seeds = {row["seed"] for row in done}
            example = group[0]
            validation_mean, validation_std = mean_std([row["validation_rmse"] for row in done])
            test_mean, test_std = mean_std([row["test_rmse"] for row in done])
            summaries.append({**{key: example[key] for key in ("model", "ablation", "display_model", "tokenizer_type", "mixing_type", "channel", "readout", "dataset", "display_dataset", "K", "r", "tau", "token_mlp_ratio", "d", "L", "parameter_count", "trainable_parameter_count", "token_mlp_parameter_count", "channel_mlp_parameter_count")},
                              "expected_seeds": "0,1,2,3,4", "successful_seeds": len(found_seeds),
                              "missing_seeds": ",".join(map(str, sorted(SEEDS - found_seeds))), "complete": found_seeds == SEEDS,
                              "validation_rmse_mean": validation_mean, "validation_rmse_std": validation_std,
                              "test_rmse_mean": test_mean, "test_rmse_std": test_std})
    summary_fields = list(summaries[0])
    write(output / "summary.csv", summaries, summary_fields)
    table = []
    for ablation, label in LABELS.items():
        row = {"model": label}
        for dataset, display in DISPLAY.items():
            summary = next(item for item in summaries if item["ablation"] == ablation and item["dataset"] == dataset)
            if summary["test_rmse_mean"] is None:
                row[display] = None
            elif summary["test_rmse_std"] is None:
                row[display] = f"{summary['test_rmse_mean']:.6g}"
            else:
                row[display] = f"{summary['test_rmse_mean']:.6g} +/- {summary['test_rmse_std']:.3g}"
        table.append(row)
    write(output / "rmse_table.csv", table, list(table[0]))
    missing = [row for row in rows if row["status"] != "completed"]
    integrity = {"expected_runs": 140, "completed_runs": len(rows) - len(missing), "missing_or_invalid_runs": len(missing),
                 "expected_summaries": 28, "complete_five_seed_summaries": sum(item["complete"] for item in summaries)}
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} runs, {len(summaries)} summaries, and {len(missing)} missing/invalid rows to {output}")


if __name__ == "__main__":
    main()
