"""Summarize isolated Hinge-Mix-TokenMLP runs and optionally compare formal HingeMix."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DISPLAY = {
    "hpcg2": "HPCG",
    "hpgmg3": "HPGMG",
    "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix",
    "raiderstream": "RaiderSTREAM",
    "stream": "STREAM",
    "cachesweep": "CacheSweep",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_token_mlp")
    parser.add_argument("--output", default="results/compare/hingemix_token_mlp")
    parser.add_argument(
        "--reference-per-seed",
        default=None,
        help="Optional formal HingeMix per_seed.csv; only rows explicitly identified as full/HingeMix are read.",
    )
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return (
        statistics.mean(values) if values else None,
        statistics.stdev(values) if len(values) > 1 else None,
    )


def record(root: Path, task: dict) -> dict:
    directory = root / task["dataset"] / f"seed_{task['seed']}"
    marker = read_json(directory / "parameter_result.json")
    attempt = Path(marker["attempt_dir"]) if marker and isinstance(marker.get("attempt_dir"), str) else None
    meta = read_json(attempt / "task.json") if attempt else None
    prediction = read_json(attempt / "prediction.json") if attempt else None
    history = read_json(attempt / "results.json") if attempt else None
    status, reason = "missing", "result has not completed"
    if marker and marker.get("config_fingerprint") != task["config_fingerprint"]:
        status, reason = "conflict", "result fingerprint mismatch"
    elif meta and meta.get("config_fingerprint") == task["config_fingerprint"]:
        if isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
            status, reason = "completed", None
        else:
            status, reason = "invalid", "test RMSE missing or invalid"
    effective = (meta or {}).get("effective_parameters", {})
    value = marker or {}
    validation = (history or {}).get("val") or {}
    return {
        "model": "hingemix_token_mlp",
        "display_model": "Hinge-Mix-TokenMLP",
        "dataset": task["dataset"],
        "display_dataset": DISPLAY[task["dataset"]],
        "seed": task["seed"],
        "K": effective.get("K"),
        "token_mlp_ratio": effective.get("token_mlp_ratio"),
        "d": effective.get("d"),
        "L": effective.get("L"),
        "r": effective.get("r"),
        "tau": effective.get("tau"),
        "validation_rmse": value.get("validation_rmse", validation.get("best_metric")),
        "test_rmse": value.get("test_rmse", (prediction or {}).get("metrics", {}).get("rmse")),
        "best_epoch": value.get("best_epoch", validation.get("best_epoch")),
        "parameter_count": value.get("parameter_count"),
        "trainable_parameter_count": value.get("trainable_parameter_count"),
        "token_mlp_parameter_count": value.get("token_mlp_parameter_count"),
        "channel_mlp_parameter_count": value.get("channel_mlp_parameter_count"),
        "status": status,
        "reason": reason,
        "attempt_dir": str(attempt) if attempt else None,
    }


def summaries(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["dataset"]), []).append(row)
    result = []
    for _, group in sorted(grouped.items()):
        completed = [row for row in group if row["status"] == "completed"]
        example = group[0]
        validation_mean, validation_std = mean_std([row["validation_rmse"] for row in completed])
        test_mean, test_std = mean_std([row["test_rmse"] for row in completed])
        result.append({
            **{key: example[key] for key in (
                "model", "display_model", "dataset", "display_dataset", "K", "token_mlp_ratio",
                "d", "L", "r", "tau", "parameter_count", "trainable_parameter_count",
                "token_mlp_parameter_count", "channel_mlp_parameter_count",
            )},
            "expected_seeds": 5,
            "successful_seeds": len(completed),
            "missing_seeds": 5 - len(completed),
            "complete": len(completed) == 5,
            "validation_rmse_mean": validation_mean,
            "validation_rmse_std": validation_std,
            "test_rmse_mean": test_mean,
            "test_rmse_std": test_std,
        })
    return result


def load_reference(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        if row.get("experiment") not in ("full", None) and row.get("display_model") != "HingeMix":
            continue
        try:
            seed = int(row["seed"])
            test_rmse = float(row["test_rmse"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "model": "hingemix_reference",
            "display_model": "HingeMix",
            "dataset": row["dataset"],
            "display_dataset": row.get("display_dataset", DISPLAY.get(row["dataset"], row["dataset"])),
            "seed": seed,
            "K": row.get("K"), "token_mlp_ratio": None, "d": row.get("d"), "L": row.get("L"),
            "r": row.get("r"), "tau": row.get("tau"),
            "validation_rmse": float(row["validation_rmse"]) if row.get("validation_rmse") else None,
            "test_rmse": test_rmse,
            "best_epoch": row.get("best_epoch"),
            "parameter_count": row.get("parameter_count"), "trainable_parameter_count": row.get("trainable_parameter_count"),
            "token_mlp_parameter_count": None, "channel_mlp_parameter_count": None,
            "status": "completed", "reason": None, "attempt_dir": row.get("attempt_dir"),
        })
    return rows


def main() -> None:
    args = parse_args()
    root, output = Path(args.root), Path(args.output)
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest.get("expected_tasks") != 35:
        raise SystemExit("No compatible HingeMix TokenMLP manifest found")
    rows = [record(root, task) for task in manifest["tasks"]]
    fields = list(rows[0])
    write_csv(output / "per_seed.csv", rows, fields)
    write_csv(output / "missing_runs.csv", [row for row in rows if row["status"] != "completed"], fields)
    summary = summaries(rows)
    summary_fields = list(summary[0])
    write_csv(output / "summary.csv", summary, summary_fields)
    long_table = []
    for item in summary:
        long_table.append({
            "model": item["display_model"], "dataset": item["display_dataset"],
            "test_rmse": item["test_rmse_mean"], "test_rmse_std": item["test_rmse_std"],
            "successful_seeds": item["successful_seeds"], "parameter_count": item["parameter_count"],
        })
    write_csv(output / "rmse_table_long.csv", long_table, list(long_table[0]))
    if args.reference_per_seed:
        reference = load_reference(Path(args.reference_per_seed))
        combined = rows + reference
        write_csv(output / "comparison_per_seed.csv", combined, fields)
        comparison_summary = summaries(combined)
        write_csv(output / "comparison_summary.csv", comparison_summary, summary_fields)
    else:
        comparison_summary = summary
    wide = []
    for model in sorted({item["display_model"] for item in comparison_summary}):
        row = {"model": model}
        for dataset in DISPLAY.values():
            item = next(
                (candidate for candidate in comparison_summary if candidate["display_model"] == model and candidate["display_dataset"] == dataset),
                None,
            )
            if item is None or item["test_rmse_mean"] is None:
                row[dataset] = None
            elif item["test_rmse_std"] is None:
                row[dataset] = f"{item['test_rmse_mean']:.6g}"
            else:
                row[dataset] = f"{item['test_rmse_mean']:.6g} +/- {item['test_rmse_std']:.3g}"
        wide.append(row)
    write_csv(output / "rmse_table.csv", wide, list(wide[0]))
    integrity = {
        "expected_runs": 35,
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "missing_or_invalid_runs": sum(row["status"] != "completed" for row in rows),
        "complete_five_seed_summaries": sum(item["complete"] for item in summary),
    }
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} runs and {len(summary)} summaries to {output}")


if __name__ == "__main__":
    main()
