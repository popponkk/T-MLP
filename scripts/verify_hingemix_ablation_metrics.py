"""Read-only RMSE/MAE verification for the formal four-way HingeMix ablation v2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
DISPLAY_DATASET = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}
CONFIGURATIONS = ("full", "token_mlp", "linear", "linear_token_mlp")
DISPLAY_CONFIGURATION = {
    "full": "full", "token_mlp": "mlp", "linear": "linear", "linear_token_mlp": "linear_mlp",
}
SPECS = {
    "full": {"tokenizer_type": "ggpl", "mixing_type": "graph"},
    "token_mlp": {"tokenizer_type": "ggpl", "mixing_type": "token_mlp"},
    "linear": {"tokenizer_type": "shared_linear", "mixing_type": "graph"},
    "linear_token_mlp": {"tokenizer_type": "shared_linear", "mixing_type": "token_mlp"},
}
SEEDS = {0, 1, 2, 3, 4}
MODEL_PARAMETERS = {
    "num_breakpoints": 8, "graph_dynamic_rank": 16, "graph_temperature": 16.0,
    "d_token": 1024, "n_layers": 1,
}
TRAINING_PARAMETERS = {"lr": 1e-5, "batch_size": 32}
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/hingemix_ablation_v2")
    parser.add_argument("--reference", default=None, help="Existing v2 summary directory; defaults beside --root.")
    parser.add_argument("--baseline-reference", default=None, help="Existing verified full HingeMix metrics directory.")
    parser.add_argument("--output-root", default="results/hingemix_ablation_metrics_verified")
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


def read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), None
    except OSError as error:
        return [], f"cannot read {path}: {error}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual: Any, expected: Any) -> bool:
    return finite(actual) and finite(expected) and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE)


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def expected_model_keys(configuration: str) -> set[str]:
    keys = {"d_token", "n_layers"}
    if SPECS[configuration]["tokenizer_type"] == "ggpl":
        keys.add("num_breakpoints")
    if SPECS[configuration]["mixing_type"] == "graph":
        keys.update({"graph_dynamic_rank", "graph_temperature"})
    return keys


def row_failure(task: dict[str, Any], task_root: Path, reason: str) -> dict[str, Any]:
    configuration = str(task.get("ablation", "unknown"))
    dataset = str(task.get("dataset", "unknown"))
    return {
        "configuration": configuration,
        "display_configuration": DISPLAY_CONFIGURATION.get(configuration, configuration),
        "dataset": dataset,
        "display_name": DISPLAY_DATASET.get(dataset, dataset),
        "seed": task.get("seed"),
        "test_rmse": None,
        "test_mae": None,
        "metric_scale": "unknown",
        "source_path": str((task_root / "parameter_result.json").resolve()),
        "attempt_path": None,
        "extraction_method": "not_extracted",
        "rmse_reference_diff": None,
        "baseline_full_rmse_diff": None,
        "baseline_full_mae_diff": None,
        "verification_status": "invalid",
        "reason": reason,
    }


def task_root(root: Path, task: dict[str, Any]) -> Path:
    return root / task["ablation"] / task["dataset"] / f"seed_{task['seed']}"


def verify_task(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    directory = task_root(root, task)
    configuration = task.get("ablation")
    dataset = task.get("dataset")
    if configuration not in CONFIGURATIONS or dataset not in DATASETS:
        return row_failure(task, directory, "task has unsupported configuration or dataset")
    if task.get("model") != "hingemix_ablation_v2" or task.get("spec") != SPECS[configuration]:
        return row_failure(task, directory, "manifest model/specification mismatch")
    if task.get("data_seed") != 42:
        return row_failure(task, directory, "data_seed is not 42")
    parameters = task.get("parameters") or {}
    if any(not close(parameters.get(key), MODEL_PARAMETERS[key]) for key in expected_model_keys(configuration)):
        return row_failure(task, directory, "manifest effective model parameters mismatch")
    if any(not close(task.get(key), value) for key, value in TRAINING_PARAMETERS.items()):
        return row_failure(task, directory, "manifest training parameters mismatch")

    marker_path = directory / "parameter_result.json"
    marker, error = read_json(marker_path)
    if error:
        return row_failure(task, directory, error)
    if marker.get("config_fingerprint") != task.get("config_fingerprint"):
        return row_failure(task, directory, "task-root marker fingerprint mismatch")
    attempt_value = marker.get("attempt_dir")
    if not isinstance(attempt_value, str):
        return row_failure(task, directory, "task-root marker has no selected attempt_dir")
    attempt = Path(attempt_value)
    metadata, error = read_json(attempt / "task.json")
    if error:
        return row_failure(task, directory, error)
    if metadata.get("config_fingerprint") != task.get("config_fingerprint"):
        return row_failure(task, directory, "selected attempt fingerprint mismatch")
    if metadata.get("ablation") != configuration or metadata.get("spec") != SPECS[configuration]:
        return row_failure(task, directory, "selected attempt configuration/specification mismatch")

    config, error = read_yaml(attempt / "final_config.yaml")
    if error:
        return row_failure(task, directory, error)
    model, training = config.get("model"), config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        return row_failure(task, directory, "selected attempt final config lacks model/training mappings")
    if model.get("ablation") != configuration or model.get("model_name") != "hingemix_ablation_v2":
        return row_failure(task, directory, "final config is not hingemix_ablation_v2 selected configuration")
    if any(not close(model.get(key), MODEL_PARAMETERS[key]) for key in expected_model_keys(configuration)):
        return row_failure(task, directory, "selected attempt final model parameters mismatch")
    if any(not close(training.get(key), value) for key, value in TRAINING_PARAMETERS.items()):
        return row_failure(task, directory, "selected attempt final training parameters mismatch")

    prediction, error = read_json(attempt / "prediction.json")
    if error:
        return row_failure(task, directory, error)
    metrics = prediction.get("metrics")
    if prediction.get("metric_name") != "rmse" or not isinstance(metrics, dict):
        return row_failure(task, directory, "prediction is not a final regression test metric record")
    rmse, mae = metrics.get("rmse"), metrics.get("mae")
    if not finite(rmse) or not finite(mae) or rmse < 0 or mae < 0:
        return row_failure(task, directory, "test RMSE/MAE is missing, non-finite, or negative")
    if mae > rmse + TOLERANCE:
        return row_failure(task, directory, "test MAE exceeds test RMSE")
    if not close(prediction.get("metric"), rmse):
        return row_failure(task, directory, "prediction metric conflicts with metrics.rmse")
    if finite(marker.get("test_rmse")) and not close(marker["test_rmse"], rmse):
        return row_failure(task, directory, "task-root marker test_rmse conflicts with prediction")
    return {
        "configuration": configuration,
        "display_configuration": DISPLAY_CONFIGURATION[configuration],
        "dataset": dataset,
        "display_name": DISPLAY_DATASET[dataset],
        "seed": task["seed"],
        "test_rmse": rmse,
        "test_mae": mae,
        "metric_scale": "original_target_scale (standardized errors multiplied by training y_std)",
        "source_path": str(marker_path.resolve()),
        "attempt_path": str(attempt.resolve()),
        "extraction_method": "direct: selected attempt prediction.json metrics.rmse + metrics.mae",
        "rmse_reference_diff": None,
        "baseline_full_rmse_diff": None,
        "baseline_full_mae_diff": None,
        "verification_status": "verified",
        "reason": "selected by existing task-root marker; manifest, attempt and final config agree",
    }


def reference_maps(reference: Path | None) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str], float], list[str]]:
    issues: list[str] = []
    per_seed: dict[tuple[str, str, int], float] = {}
    summary: dict[tuple[str, str], float] = {}
    if reference is None:
        return per_seed, summary, ["Existing v2 summary directory was not found; no prior CSV cross-check performed."]
    rows, error = read_csv(reference / "per_seed.csv")
    if error:
        issues.append(error)
    else:
        for row in rows:
            value, seed = number(row.get("test_rmse")), number(row.get("seed"))
            if value is not None and seed is not None:
                per_seed[(row.get("ablation", ""), row.get("dataset", ""), int(seed))] = value
    rows, error = read_csv(reference / "summary.csv")
    if error:
        issues.append(error)
    else:
        for row in rows:
            value = number(row.get("test_rmse_mean"))
            if value is not None:
                summary[(row.get("ablation", ""), row.get("dataset", ""))] = value
    return per_seed, summary, issues


def baseline_map(directory: Path | None) -> tuple[dict[tuple[str, int], tuple[float, float]], list[str]]:
    values: dict[tuple[str, int], tuple[float, float]] = {}
    if directory is None:
        return values, ["Verified baseline metrics directory was not found; full-reference cross-check not performed."]
    rows, error = read_csv(directory / "per_seed_metrics.csv")
    if error:
        return values, [error]
    for row in rows:
        if row.get("configuration") != "full" or row.get("verification_status") != "verified":
            continue
        rmse, mae, seed = number(row.get("test_rmse")), number(row.get("test_mae")), number(row.get("seed"))
        if rmse is not None and mae is not None and seed is not None:
            values[(row.get("dataset", ""), int(seed))] = (rmse, mae)
    return values, []


def compare_references(rows: list[dict[str, Any]], per_seed: dict[tuple[str, str, int], float], baseline: dict[tuple[str, int], tuple[float, float]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        if row["verification_status"] != "verified":
            issues.append({**row, "issue_type": "invalid_record"})
            continue
        key = (row["configuration"], row["dataset"], int(row["seed"]))
        if key not in per_seed:
            row["verification_status"] = "reference_missing"
            row["reason"] += "; existing v2 per_seed reference missing"
            issues.append({**row, "issue_type": "reference_missing"})
        else:
            difference = abs(float(row["test_rmse"]) - per_seed[key])
            row["rmse_reference_diff"] = difference
            if difference > TOLERANCE:
                row["verification_status"] = "rmse_reference_mismatch"
                row["reason"] += "; existing v2 per_seed RMSE mismatch"
                issues.append({**row, "issue_type": "per_seed_rmse_mismatch"})
        if row["configuration"] == "full" and baseline:
            baseline_values = baseline.get((row["dataset"], int(row["seed"])))
            if baseline_values is None:
                row["verification_status"] = "baseline_reference_missing"
                row["reason"] += "; verified baseline full reference missing"
                issues.append({**row, "issue_type": "baseline_reference_missing"})
            else:
                rmse_diff, mae_diff = abs(float(row["test_rmse"]) - baseline_values[0]), abs(float(row["test_mae"]) - baseline_values[1])
                row["baseline_full_rmse_diff"], row["baseline_full_mae_diff"] = rmse_diff, mae_diff
                if rmse_diff > TOLERANCE or mae_diff > TOLERANCE:
                    row["verification_status"] = "baseline_reference_mismatch"
                    row["reason"] += "; verified baseline full metric mismatch"
                    issues.append({**row, "issue_type": "baseline_full_metric_mismatch"})
    return issues


def summarize(rows: list[dict[str, Any]], reference_summary: dict[tuple[str, str], float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, issues = [], []
    for configuration in CONFIGURATIONS:
        for dataset in DATASETS:
            group = [row for row in rows if row["configuration"] == configuration and row["dataset"] == dataset]
            # A missing or conflicting legacy reference does not make the selected
            # final prediction invalid. It is reported separately below.
            valid = [row for row in group if finite(row.get("test_rmse")) and finite(row.get("test_mae")) and row["verification_status"] != "invalid"]
            seeds = {int(row["seed"]) for row in valid if isinstance(row.get("seed"), int)}
            complete = len(valid) == 5 and seeds == SEEDS
            rmse_mean = statistics.mean(float(row["test_rmse"]) for row in valid) if complete else None
            mae_mean = statistics.mean(float(row["test_mae"]) for row in valid) if complete else None
            reference = reference_summary.get((configuration, dataset))
            diff = abs(rmse_mean - reference) if rmse_mean is not None and reference is not None else None
            status = "verified" if complete else "incomplete"
            reference_statuses = {row["verification_status"] for row in valid}
            if complete and any("mismatch" in item for item in reference_statuses):
                status = "per_seed_reference_mismatch"
            elif complete and any(item == "reference_missing" for item in reference_statuses):
                status = "per_seed_reference_missing"
            elif complete and any(item == "baseline_reference_missing" for item in reference_statuses):
                status = "baseline_reference_missing"
            elif complete and any(item == "baseline_reference_mismatch" for item in reference_statuses):
                status = "baseline_reference_mismatch"
            elif complete and reference is None:
                status = "summary_reference_missing"
            elif diff is not None and diff > TOLERANCE:
                status = "summary_rmse_mismatch"
            item = {
                "configuration": configuration,
                "display_configuration": DISPLAY_CONFIGURATION[configuration],
                "dataset": dataset,
                "display_name": DISPLAY_DATASET[dataset],
                "n_valid_rmse": len(valid),
                "n_valid_mae": len(valid),
                "valid_seeds": ",".join(map(str, sorted(seeds))),
                "rmse_mean": rmse_mean,
                "mae_mean": mae_mean,
                "rmse_reference_mean": reference,
                "rmse_reference_diff": diff,
                "status": status,
            }
            summary.append(item)
            if status != "verified":
                issues.append({**item, "issue_type": status, "reason": "five-seed metric group is incomplete or disagrees with prior summary"})
    return summary, issues


def table_rows(summary: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        row = {"dataset": dataset, "display_name": DISPLAY_DATASET[dataset]}
        for configuration in CONFIGURATIONS:
            item = next(value for value in summary if value["dataset"] == dataset and value["configuration"] == configuration)
            row[DISPLAY_CONFIGURATION[configuration]] = item[f"{metric}_mean"] if item["status"] == "verified" else None
        rows.append(row)
    return rows


def table_markdown(summary: list[dict[str, Any]], metric: str) -> list[str]:
    heading = "RMSE" if metric == "rmse" else "MAE"
    lines = [f"## {heading} table", "", "| Dataset | full | mlp | linear | linear_mlp |", "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        records = [next(value for value in summary if value["dataset"] == dataset and value["configuration"] == configuration) for configuration in CONFIGURATIONS]
        candidates = [float(value[f"{metric}_mean"]) for value in records if value["status"] == "verified"]
        best = min(candidates) if candidates else None
        cells = []
        for value in records:
            raw = value[f"{metric}_mean"]
            if value["status"] != "verified" or raw is None:
                cells.append("待核实")
            else:
                formatted = f"{raw:.4f}"
                cells.append(f"**{formatted}**" if best is not None and abs(float(raw) - best) <= TOLERANCE else formatted)
        lines.append(f"| {DISPLAY_DATASET[dataset]} | " + " | ".join(cells) + " |")
    return lines


def report(output: Path, root: Path, reference: Path | None, baseline: Path | None, rows: list[dict[str, Any]], summary: list[dict[str, Any]], reference_issues: list[str]) -> None:
    verified = sum(row["verification_status"] == "verified" for row in rows)
    complete = sum(row["status"] == "verified" for row in summary)
    lines = [
        "# HingeMix Formal Ablation v2 RMSE/MAE Verification",
        "",
        f"- Result root: `{root.resolve()}`",
        f"- Existing v2 reference: `{reference.resolve()}`" if reference else "- Existing v2 reference: unavailable",
        f"- Verified full baseline reference: `{baseline.resolve()}`" if baseline else "- Verified full baseline reference: unavailable",
        "- Scope: only `hingemix_ablation_v2` manifest tasks `full`, `token_mlp`, `linear`, `linear_token_mlp`.",
        "- Selected attempt: task-root `parameter_result.json.attempt_dir`; no selection by RMSE or MAE.",
        "- Metric source: selected attempt `prediction.json.metrics.rmse` and `.mae` from final test evaluation.",
        "- Scale: original target scale; standardized error is multiplied by the training-split `y_std`.",
        "- Checkpoint selection: validation RMSE selects `best.bin`; the final test metrics are produced after reloading it.",
        f"- Verified per-seed metrics: {verified}/140.",
        f"- Complete configuration × dataset groups: {complete}/28.",
        "",
        "## Cross-check status",
        "",
        "| Configuration | Dataset | Valid RMSE | Valid MAE | RMSE reference difference | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in summary:
        diff = "待核实" if item["rmse_reference_diff"] is None else f"{item['rmse_reference_diff']:.3g}"
        lines.append(f"| {item['display_configuration']} | {item['display_name']} | {item['n_valid_rmse']} | {item['n_valid_mae']} | {diff} | {item['status']} |")
    if reference_issues:
        lines += ["", "## Reference availability notes", ""] + [f"- {item}" for item in reference_issues]
    lines += [""] + table_markdown(summary, "rmse") + [""] + table_markdown(summary, "mae") + [
        "",
        "## Implementation evidence",
        "",
        "- `models/abstract.py:652-680`: validation metric controls best-checkpoint selection.",
        "- `main.py:267-278`: reloads the best checkpoint and writes final test `prediction.json`.",
        "- `utils/metrics.py:14-29,71-77`: RMSE and MAE use the same training-split `y_std` to restore original scale.",
        "- `scripts/run_hingemix_ablation_v2.py:107-121`: selected completed result requires marker/attempt fingerprint agreement and final test RMSE.",
    ]
    (output / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_dir(root: Path) -> Path:
    if not root.exists():
        return root
    return root / dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run(root: Path, output_root: Path, reference: Path | None, baseline: Path | None) -> Path:
    manifest, error = read_json(root / "manifest.json")
    if error:
        raise SystemExit(error)
    if manifest.get("experiment") != "hingemix_ablation_v2" or manifest.get("expected_tasks") != 140:
        raise SystemExit("No compatible 140-task hingemix_ablation_v2 manifest found")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 140:
        raise SystemExit("Manifest does not contain exactly 140 tasks")
    identities = [(task.get("ablation"), task.get("dataset"), task.get("seed")) for task in tasks]
    if len(set(identities)) != 140:
        raise SystemExit("Manifest contains duplicate configuration/dataset/seed identities")
    rows = [verify_task(root, task) for task in sorted(tasks, key=lambda value: (CONFIGURATIONS.index(value["ablation"]), DATASETS.index(value["dataset"]), value["seed"]))]
    per_seed_reference, summary_reference, reference_issues = reference_maps(reference)
    baseline_values, baseline_issues = baseline_map(baseline)
    reference_issues.extend(baseline_issues)
    issues = compare_references(rows, per_seed_reference, baseline_values)
    summary, summary_issues = summarize(rows, summary_reference)
    issues.extend(summary_issues)
    output = output_dir(output_root)
    output.mkdir(parents=True, exist_ok=False)
    fields = [
        "configuration", "display_configuration", "dataset", "display_name", "seed", "test_rmse", "test_mae",
        "metric_scale", "source_path", "attempt_path", "extraction_method", "rmse_reference_diff",
        "baseline_full_rmse_diff", "baseline_full_mae_diff", "verification_status", "reason",
    ]
    write_csv(output / "per_seed_metrics.csv", rows, fields)
    write_csv(output / "summary_metrics.csv", summary, list(summary[0]))
    write_csv(output / "rmse_table.csv", table_rows(summary, "rmse"), ["dataset", "display_name", "full", "mlp", "linear", "linear_mlp"])
    write_csv(output / "mae_table.csv", table_rows(summary, "mae"), ["dataset", "display_name", "full", "mlp", "linear", "linear_mlp"])
    issue_fields = fields + ["issue_type"]
    write_csv(output / "missing_or_mismatched.csv", issues, issue_fields)
    report(output, root, reference, baseline, rows, summary, reference_issues)
    print(f"Wrote {len(rows)} per-seed rows, {len(summary)} summaries, and {len(issues)} issues to {output}")
    return output


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "results"
        task = {
            "model": "hingemix_ablation_v2", "ablation": "linear_token_mlp",
            "spec": SPECS["linear_token_mlp"], "dataset": "hpcg2", "seed": 0, "data_seed": 42,
            "parameters": {**MODEL_PARAMETERS, "token_mlp_ratio": 2.0}, "lr": 1e-5, "batch_size": 32,
            "config_fingerprint": "test",
        }
        directory = task_root(root, task)
        attempt = directory / "attempts" / "attempt_001"
        attempt.mkdir(parents=True)
        (directory / "parameter_result.json").write_text(json.dumps({"config_fingerprint": "test", "attempt_dir": str(attempt), "test_rmse": 1.0}), encoding="utf-8")
        (attempt / "task.json").write_text(json.dumps(task), encoding="utf-8")
        final_model = {"model_name": "hingemix_ablation_v2", "ablation": "linear_token_mlp", "d_token": 1024, "n_layers": 1, "token_mlp_ratio": 2.0}
        (attempt / "final_config.yaml").write_text(yaml.safe_dump({"model": final_model, "training": TRAINING_PARAMETERS}), encoding="utf-8")
        (attempt / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.0, "metrics": {"rmse": 1.0, "mae": 0.5}}), encoding="utf-8")
        assert verify_task(root, task)["verification_status"] == "verified"
        (attempt / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.0, "metrics": {"rmse": 1.0, "mae": 1.1}}), encoding="utf-8")
        assert verify_task(root, task)["verification_status"] == "invalid"
    print("Self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    root = Path(args.root)
    reference = Path(args.reference) if args.reference else root.parent / "hingemix_ablation_v2_summary"
    baseline = Path(args.baseline_reference) if args.baseline_reference else root.parent / "hingemix_baseline_metrics_verified"
    run(root, Path(args.output_root), reference if reference.is_dir() else None, baseline if baseline.is_dir() else None)


if __name__ == "__main__":
    main()
