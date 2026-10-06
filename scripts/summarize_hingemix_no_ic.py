#!/usr/bin/env python3
"""Summarize only the isolated RaiderSTREAM/CacheSweep no-IC experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DISPLAY = {"raiderstream_no_ic": "RaiderSTREAM (no instructions/cycles)", "cachesweep_no_ic": "CacheSweep (no instructions/cycles)"}
LABELS = {"full": "HingeMix", "no_graph": "HingeMix-no_graph", "linear": "HingeMix-linear", "linear_no_graph": "HingeMix-linear_no_graph",
          "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost", "excel-former": "ExcelFormer", "ft-transformer": "FT-Transformer", "node": "NODE", "mlp": "MLP", "autoint": "AutoInt", "tabm": "TabM"}


def read(path: Path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return None


def write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or (list(rows[0]) if rows else [])); writer.writeheader(); writer.writerows(rows)


def stats(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_no_ic")
    parser.add_argument("--output", default="results/compare/hingemix_no_ic")
    args = parser.parse_args(); root, output = Path(args.root), Path(args.output)
    manifest = read(root / "manifest.json")
    if not manifest: raise SystemExit("Missing no-IC manifest")
    runs = []
    for task in manifest["tasks"]:
        directory = root / task["experiment"] / task["dataset"] / f"seed_{task['seed']}"; marker = read(directory / "parameter_result.json")
        attempt = Path(marker["attempt_dir"]) if marker and isinstance(marker.get("attempt_dir"), str) else None
        done = bool(marker and marker.get("config_fingerprint") == task["config_fingerprint"] and isinstance(marker.get("test_rmse"), (int, float)))
        runs.append({"experiment": task["experiment"], "display_model": LABELS[task["experiment"]], "dataset": task["dataset"], "display_dataset": DISPLAY[task["dataset"]], "seed": task["seed"],
                     "test_rmse": marker.get("test_rmse") if marker else None, "validation_rmse": marker.get("validation_rmse") if marker else None,
                     "status": "completed" if done else "missing_or_invalid", "attempt_dir": str(attempt) if attempt else "", "result_marker": str(directory / "parameter_result.json")})
    write(output / "per_seed.csv", runs)
    missing = [row for row in runs if row["status"] != "completed"]; write(output / "missing_runs.csv", missing, list(runs[0]))
    summaries = []
    for experiment in LABELS:
        for dataset in DISPLAY:
            group = [row for row in runs if row["experiment"] == experiment and row["dataset"] == dataset and row["status"] == "completed"]
            test_mean, test_std = stats([row["test_rmse"] for row in group]); val_mean, val_std = stats([row["validation_rmse"] for row in group])
            summaries.append({"experiment": experiment, "display_model": LABELS[experiment], "dataset": dataset, "display_dataset": DISPLAY[dataset], "successful_seeds": len(group), "expected_seeds": 5,
                              "complete": len(group) == 5, "test_rmse_mean": test_mean, "test_rmse_std": test_std, "validation_rmse_mean": val_mean, "validation_rmse_std": val_std})
    write(output / "summary.csv", summaries)
    table = []
    for experiment in LABELS:
        row = {"model": LABELS[experiment]}
        for dataset, display in DISPLAY.items():
            item = next(summary for summary in summaries if summary["experiment"] == experiment and summary["dataset"] == dataset)
            row[display] = "" if item["test_rmse_mean"] is None else f"{item['test_rmse_mean']:.6g}" + (f" +/- {item['test_rmse_std']:.3g}" if item["test_rmse_std"] is not None else "")
        table.append(row)
    write(output / "rmse_table.csv", table)
    references = [{"reference_experiment": "full_ablation", "source_experiment": "full", "reason": "Same HingeMix implementation/configuration; no duplicate full-ablation training was scheduled.", "run_count": 10}]
    write(output / "full_reuse.csv", references)
    (output / "integrity.json").write_text(json.dumps({"expected_physical_runs": 130, "completed_runs": len(runs) - len(missing), "missing_or_invalid_runs": len(missing), "full_ablation_reused_runs": 10}, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} physical runs, {len(summaries)} summaries, and {len(missing)} missing rows to {output}")


if __name__ == "__main__": main()
