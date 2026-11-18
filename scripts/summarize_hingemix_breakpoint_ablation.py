#!/usr/bin/env python3
"""Summarize verified HingeMix breakpoint-initialization comparison results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


DATASETS = (
    ("hpcg2", "HPCG"), ("hpgmg3", "HPGMG"), ("ramspeed", "RAMspeed"),
    ("mix_with_five_datasets161", "mix"), ("raiderstream", "RaiderSTREAM"),
    ("stream", "STREAM"), ("cachesweep", "CacheSweep"),
)
STRATEGIES = (("gbdt", "GBDT"), ("quantile", "Quantile"), ("uniform", "Uniform"))
SEEDS = {0, 1, 2, 3, 4}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/hingemix_breakpoint_ablation")
    parser.add_argument("--output-root", default="results/hingemix_breakpoint_ablation_summary")
    return parser.parse_args()


def selection(manifest: dict[str, Any], root: Path, task: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    result = read_json(root / task["strategy"] / task["dataset"] / f"seed_{task['training_seed']}" / "parameter_result.json")
    if not result or result.get("config_fingerprint") != task.get("config_fingerprint"):
        return None, "missing or conflicting selected parameter_result.json"
    attempt = Path(result.get("attempt_dir", ""))
    metadata = read_json(attempt / "task.json")
    prediction = read_json(attempt / "prediction.json")
    diagnostics = read_json(attempt / "breakpoint_diagnostics.json")
    if not metadata or metadata.get("config_fingerprint") != task.get("config_fingerprint"):
        return None, "attempt task fingerprint mismatch"
    metrics = (prediction or {}).get("metrics", {})
    if not all(finite_number(metrics.get(metric)) for metric in ("rmse", "mae", "r2")):
        return None, "final test metric missing or non-finite"
    if not isinstance(diagnostics, dict):
        return None, "breakpoint diagnostics missing"
    if not (attempt / "test_prediction_trace.npz").is_file():
        return None, "paired final-test prediction trace missing"
    return {"result": result, "attempt": attempt, "metrics": metrics, "diagnostics": diagnostics}, None


def format_metric(item: dict[str, Any] | None, best: float | None, second: float | None) -> str:
    if item is None or item["status"] != "complete":
        return "pending verification"
    value, std = item["rmse_mean"], item["rmse_std"]
    text = f"{value:.4f} +/- {std:.4f}"
    if best is not None and value == best:
        return f"**{text}**"
    if second is not None and value == second:
        return f"<u>{text}</u>"
    return text


def save_plot(output: Path, dataset: str, display: str, rows: list[dict[str, Any]]) -> str | None:
    complete = [row for row in rows if row["status"] == "complete"]
    if not complete:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = {"gbdt": "#1b6ca8", "quantile": "#e17c05", "uniform": "#3a913f"}
    labels = [row["display_strategy"] for row in complete]
    values = [row["rmse_mean"] for row in complete]
    errors = [row["rmse_std"] for row in complete]
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    positions = list(range(len(complete)))
    axis.bar(positions, values, yerr=errors, capsize=4, color=[colors[row["strategy"]] for row in complete])
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Test RMSE (original target scale)")
    axis.set_title(f"HingeMix breakpoint initialization: {display}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    base = output / f"rmse_{dataset}"
    figure.savefig(base.with_suffix(".png"), dpi=180)
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)
    return str(base.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest.get("experiment") != "hingemix_breakpoint_ablation":
        raise SystemExit(f"No compatible manifest at {root / 'manifest.json'}")
    output_base = Path(args.output_root).resolve()
    output = output_base / f"summary_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    per_seed: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    expected = {(strategy, dataset, seed) for strategy, _ in STRATEGIES for dataset, _ in DATASETS for seed in SEEDS}
    task_by_key = {(task.get("strategy"), task.get("dataset"), task.get("training_seed")): task for task in manifest.get("tasks", [])}
    for strategy, display_strategy in STRATEGIES:
        for dataset, display_name in DATASETS:
            for seed in sorted(SEEDS):
                task = task_by_key.get((strategy, dataset, seed))
                base = {"configuration": strategy, "display_configuration": display_strategy, "dataset": dataset, "display_name": display_name, "seed": seed}
                if task is None:
                    per_seed.append({**base, "verification_status": "missing", "reason": "task absent from manifest"})
                    missing.append({**base, "kind": "missing_task", "reason": "task absent from manifest"})
                    continue
                chosen, reason = selection(manifest, root, task)
                if chosen is None:
                    per_seed.append({**base, "config_fingerprint": task["config_fingerprint"], "verification_status": "missing", "reason": reason})
                    missing.append({**base, "kind": "missing_or_invalid", "reason": reason})
                    continue
                metrics, diag, attempt = chosen["metrics"], chosen["diagnostics"], chosen["attempt"]
                row = {
                    **base, "config_fingerprint": task["config_fingerprint"], "test_rmse": metrics["rmse"], "test_mae": metrics["mae"], "test_r2": metrics["r2"],
                    "metric_scale": "original_target_scale", "attempt_path": str(attempt), "prediction_path": str(attempt / "prediction.json"),
                    "test_prediction_trace_path": str(attempt / "test_prediction_trace.npz"),
                    "breakpoint_diagnostics_path": str(attempt / "breakpoint_diagnostics.json"), "best_epoch": chosen["result"].get("best_epoch"),
                    "validation_rmse": chosen["result"].get("validation_rmse"), "verification_status": "verified", "reason": "",
                }
                per_seed.append(row)
                final = diag.get("final_checkpoint_breakpoints", {})
                diagnostics_rows.append({
                    **base, "attempt_path": str(attempt), "breakpoint_strategy": diag.get("strategy"),
                    "breakpoint_slots": diag.get("num_breakpoint_slots"), "learnable_breakpoints": diag.get("learnable_breakpoints"),
                    "generation_seconds": diag.get("generation_seconds"), "constant_feature_indices": json.dumps(diag.get("constant_feature_indices", [])),
                    "requested_unique_breakpoint_counts": json.dumps(diag.get("requested_unique_breakpoint_counts", [])),
                    "effective_unique_breakpoint_counts": json.dumps(diag.get("effective_unique_breakpoint_counts", [])),
                    "final_unique_breakpoint_counts": json.dumps(final.get("final_unique_breakpoint_counts", [])),
                    "gbdt_called": (diag.get("gbdt") or {}).get("called"),
                    "gbdt_raw_candidate_threshold_counts": json.dumps((diag.get("gbdt") or {}).get("raw_candidate_threshold_counts", [])),
                    "gbdt_candidate_threshold_counts": json.dumps((diag.get("gbdt") or {}).get("candidate_threshold_counts", [])),
                    "initial_non_breakpoint_state_sha256": diag.get("non_breakpoint_state_sha256"),
                    "torch_rng_sha256_before_dataloader": diag.get("torch_rng_sha256_before_dataloader"),
                    "feature_name_source": diag.get("feature_name_source"),
                })
    for dataset, display_name in DATASETS:
        for seed in sorted(SEEDS):
            matched = [row for row in diagnostics_rows if row["dataset"] == dataset and row["seed"] == seed]
            hashes = {row.get("initial_non_breakpoint_state_sha256") for row in matched}
            rng_hashes = {row.get("torch_rng_sha256_before_dataloader") for row in matched}
            status = "verified" if len(matched) == 3 and len(hashes) == 1 and len(rng_hashes) == 1 and None not in hashes and None not in rng_hashes else "incomplete_or_mismatch"
            pairing_rows.append({
                "dataset": dataset, "display_name": display_name, "seed": seed,
                "n_strategies": len(matched), "strategies": ",".join(sorted(row["configuration"] for row in matched)),
                "non_breakpoint_state_hashes": ";".join(sorted(str(value) for value in hashes)),
                "pre_dataloader_rng_hashes": ";".join(sorted(str(value) for value in rng_hashes)),
                "status": status,
            })
    summary: list[dict[str, Any]] = []
    for strategy, display_strategy in STRATEGIES:
        for dataset, display_name in DATASETS:
            rows = [row for row in per_seed if row["configuration"] == strategy and row["dataset"] == dataset and row["verification_status"] == "verified"]
            seed_set = {row["seed"] for row in rows}
            metrics = {key: [float(row[key]) for row in rows] for key in ("test_rmse", "test_mae", "test_r2")}
            rmse_mean, rmse_std = mean_std(metrics["test_rmse"])
            mae_mean, mae_std = mean_std(metrics["test_mae"])
            r2_mean, r2_std = mean_std(metrics["test_r2"])
            status = "complete" if seed_set == SEEDS else "incomplete"
            summary.append({
                "strategy": strategy, "display_strategy": display_strategy, "dataset": dataset, "display_name": display_name,
                "n_valid_seeds": len(seed_set), "training_seeds": ",".join(map(str, sorted(seed_set))), "status": status,
                "rmse_mean": rmse_mean if status == "complete" else None, "rmse_std": rmse_std if status == "complete" else None,
                "mae_mean": mae_mean if status == "complete" else None, "mae_std": mae_std if status == "complete" else None,
                "r2_mean": r2_mean if status == "complete" else None, "r2_std": r2_std if status == "complete" else None,
                "relative_rmse_vs_gbdt_percent": None,
            })
    for dataset, _ in DATASETS:
        base = next((row for row in summary if row["dataset"] == dataset and row["strategy"] == "gbdt" and row["status"] == "complete"), None)
        for row in (item for item in summary if item["dataset"] == dataset):
            if base and row["status"] == "complete" and base["rmse_mean"] != 0:
                row["relative_rmse_vs_gbdt_percent"] = 100 * (row["rmse_mean"] - base["rmse_mean"]) / base["rmse_mean"]
    per_fields = ["configuration", "display_configuration", "dataset", "display_name", "seed", "test_rmse", "test_mae", "test_r2", "metric_scale", "attempt_path", "prediction_path", "test_prediction_trace_path", "breakpoint_diagnostics_path", "best_epoch", "validation_rmse", "config_fingerprint", "verification_status", "reason"]
    summary_fields = ["strategy", "display_strategy", "dataset", "display_name", "n_valid_seeds", "training_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std", "relative_rmse_vs_gbdt_percent", "status"]
    write_csv(output / "per_seed_results.csv", per_seed, per_fields)
    write_csv(output / "summary_metrics.csv", summary, summary_fields)
    write_csv(output / "breakpoint_diagnostics.csv", diagnostics_rows, list(diagnostics_rows[0]) if diagnostics_rows else ["configuration", "dataset", "seed"])
    write_csv(output / "initialization_pairing_checks.csv", pairing_rows, list(pairing_rows[0]))
    write_csv(output / "missing_or_mismatched.csv", missing, ["configuration", "display_configuration", "dataset", "display_name", "seed", "kind", "reason"])
    for metric, file_name in (("rmse", "rmse_table.csv"), ("mae", "mae_table.csv"), ("r2", "r2_table.csv")):
        rows = []
        for dataset, display_name in DATASETS:
            item = {"dataset": dataset, "display_name": display_name}
            for strategy, _ in STRATEGIES:
                summary_row = next(row for row in summary if row["dataset"] == dataset and row["strategy"] == strategy)
                item[strategy] = summary_row[f"{metric}_mean"] if summary_row["status"] == "complete" else None
                item[f"{strategy}_std"] = summary_row[f"{metric}_std"] if summary_row["status"] == "complete" else None
            rows.append(item)
        write_csv(output / file_name, rows, ["dataset", "display_name"] + [value for strategy, _ in STRATEGIES for value in (strategy, f"{strategy}_std")])
    plot_paths = []
    for dataset, display in DATASETS:
        path = save_plot(output, dataset, display, [row for row in summary if row["dataset"] == dataset])
        if path:
            plot_paths.append(path)
    report = [
        "# HingeMix Breakpoint Initialization Comparison",
        "",
        f"- Input root: `{root}`",
        "- Each result is the validation-selected `best.bin` final test evaluation.",
        "- Metrics are on the original target scale; RMSE and MAE are restored using the training-split target standard deviation.",
        "- Complete groups require exactly training seeds 0, 1, 2, 3, 4. Means and sample standard deviations use `ddof=1`.",
        "- `gbdt` retains the formal extraction/fallback algorithm; `quantile` and `uniform` use train-only transformed numeric features.",
        f"- Paired initialization/order checks verified: {sum(row['status'] == 'verified' for row in pairing_rows)}/{len(pairing_rows)} dataset-seed groups.",
        "",
        "## RMSE",
        "",
        "| Strategy | " + " | ".join(display for _, display in DATASETS) + " |",
        "|---|" + "---|" * len(DATASETS),
    ]
    for strategy, display_strategy in STRATEGIES:
        cells = []
        for dataset, _ in DATASETS:
            choices = [row for row in summary if row["dataset"] == dataset and row["status"] == "complete"]
            ordered = sorted({row["rmse_mean"] for row in choices})
            item = next(row for row in summary if row["dataset"] == dataset and row["strategy"] == strategy)
            cells.append(format_metric(item, ordered[0] if ordered else None, ordered[1] if len(ordered) > 1 else None))
        report.append("| " + display_strategy + " | " + " | ".join(cells) + " |")
    report.extend(["", "**MAE**", "", "| Strategy | " + " | ".join(display for _, display in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)])
    for strategy, display_strategy in STRATEGIES:
        cells = []
        for dataset, _ in DATASETS:
            item = next(row for row in summary if row["dataset"] == dataset and row["strategy"] == strategy)
            cells.append("pending verification" if item["status"] != "complete" else f"{item['mae_mean']:.4f} +/- {item['mae_std']:.4f}")
        report.append("| " + display_strategy + " | " + " | ".join(cells) + " |")
    (output / "verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    complete = sum(row["status"] == "complete" for row in summary)
    write_json(output / "integrity.json", {
        "expected_formal_tasks": len(expected), "manifest_tasks": len(manifest.get("tasks", [])),
        "verified_per_seed_records": sum(row["verification_status"] == "verified" for row in per_seed),
        "complete_strategy_dataset_groups": complete, "expected_strategy_dataset_groups": 21,
        "plots_written": len(plot_paths), "smoke": bool(manifest.get("smoke")),
        "initialization_pairing_verified": sum(row["status"] == "verified" for row in pairing_rows),
        "initialization_pairing_groups": len(pairing_rows),
    })
    print(f"Wrote {len(summary)} strategy-dataset summaries to {output}")


if __name__ == "__main__":
    main()
