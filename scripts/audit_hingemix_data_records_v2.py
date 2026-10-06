#!/usr/bin/env python3
"""Targeted, read-only follow-up audit for HingeMix experiment evidence.

Unlike the first audit, this script consumes its ``experiment_records.csv`` and
only opens those run directories plus the two specified HingeMix result trees.
It never invokes training, imports a model, changes data, or writes below results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_hingemix_data_records import (
    BASELINES, DATASETS, EXPECTED, DISPLAY_DATASET, array_info, dataset_dir,
    finite, flatten_json, load_yaml_scalars, now, read_json, sha256, write_csv,
)

try:
    import numpy as np
except ImportError:
    np = None


LINEAGE_KEYS = {"reused_from", "source_run", "source_path", "source_result", "copied_from", "origin"}
PARAMETERS = ("num_breakpoints", "graph_dynamic_rank", "graph_temperature", "d_token", "n_layers", "lr", "batch_size")
IPC_INSTRUCTIONS = {"instructions", "instruction", "total_instructions", "total_instruction"}
IPC_CYCLES = {"cycles", "cycle", "total_cycles", "total_cycle"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing first-audit record file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def json_sources(task_root: Path, attempt: Path | None, issues: list[dict[str, str]]) -> list[tuple[Path, dict[str, Any]]]:
    paths = []
    for directory in (task_root, attempt, task_root.parent, task_root.parent.parent):
        if directory is None:
            continue
        for name in ("parameter_result.json", "parameter_task.json", "task.json", "run.json", "config.json"):
            path = directory / name
            if path.is_file() and path not in paths:
                paths.append(path)
    records = []
    for path in paths:
        content = read_json(path, issues)
        if isinstance(content, dict):
            records.append((path, content))
    return records


def configuration_sources(task_root: Path, attempt: Path | None, issues: list[dict[str, str]]) -> list[tuple[Path, dict[str, Any]]]:
    records = json_sources(task_root, attempt, issues)
    for directory in (task_root, attempt):
        if directory is None:
            continue
        for name in ("final_config.yaml", "parameter_config.yaml", "configs.yaml", "data_config.yaml", "config.yaml"):
            path = directory / name
            if path.is_file():
                records.append((path, load_yaml_scalars(path, issues)))
    return records


def parameter_evidence(records: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    values: dict[str, list[tuple[str, Any]]] = {name: [] for name in PARAMETERS}
    lineage: list[str] = []
    for path, document in records:
        for field, value in flatten_json(document).items():
            leaf = field.rsplit(".", 1)[-1]
            if leaf in values and isinstance(value, (int, float, str)) and not isinstance(value, bool):
                values[leaf].append((f"{path}:{field}", value))
            if leaf.lower() in LINEAGE_KEYS and value not in (None, "", [], {}):
                lineage.append(f"{path}:{field}={value}")
    resolved, sources, conflicts = {}, {}, []
    for name, candidates in values.items():
        normalized = {str(value) for _, value in candidates}
        if not candidates:
            resolved[name], sources[name] = "", ""
        elif len(normalized) == 1:
            resolved[name] = candidates[0][1]
            sources[name] = ";".join(source for source, _ in candidates)
        else:
            resolved[name] = "conflict"
            sources[name] = ";".join(f"{source}={value}" for source, value in candidates)
            conflicts.append(name)
    return resolved, sources, lineage + (["parameter_conflicts=" + ",".join(conflicts)] if conflicts else [])


def selected_attempt(task_root: Path, issues: list[dict[str, str]]) -> tuple[dict[str, Any] | None, Path | None, list[Path]]:
    marker = read_json(task_root / "parameter_result.json", issues)
    attempts = sorted(path for path in (task_root / "attempts").glob("attempt_*") if path.is_dir()) if (task_root / "attempts").is_dir() else []
    if not isinstance(marker, dict):
        return None, None, attempts
    attempt_value = marker.get("attempt_dir")
    attempt = Path(attempt_value) if isinstance(attempt_value, str) else None
    return marker, attempt, attempts


def test_rmse(attempt: Path | None, issues: list[dict[str, str]]) -> tuple[str, str]:
    if attempt is None:
        return "", "no_selected_attempt"
    prediction = read_json(attempt / "prediction.json", issues)
    if not isinstance(prediction, dict):
        return "", "prediction_missing_or_invalid"
    candidates = []
    if prediction.get("metric_name") == "rmse" and finite(prediction.get("metric")):
        candidates.append((prediction["metric"], "prediction.json:metric"))
    metrics = prediction.get("metrics")
    if isinstance(metrics, dict) and finite(metrics.get("rmse")):
        candidates.append((metrics["rmse"], "prediction.json:metrics.rmse"))
    if not candidates:
        return "", "test_rmse_not_found"
    if len({str(value) for value, _ in candidates}) != 1:
        return "conflict", ";".join(source for _, source in candidates)
    return str(candidates[0][0]), ";".join(source for _, source in candidates)


def lineage_for_tree(label: str, root: Path, issues: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    rows = []
    for marker_path in sorted(root.rglob("parameter_result.json")):
        task_root = marker_path.parent
        # Attempt markers are evidence only; task-root marker is the logical task.
        if "attempts" in task_root.parts:
            continue
        marker, attempt, attempts = selected_attempt(task_root, issues)
        config_records = configuration_sources(task_root, attempt, issues)
        resolved, sources, lineage = parameter_evidence(config_records)
        valid = bool(marker and attempt and attempt.is_dir())
        rmse, rmse_source = test_rmse(attempt, issues)
        prediction_files = list(task_root.rglob("prediction.json"))
        result_files = list(task_root.rglob("results.json"))
        task_json = read_json(attempt / "task.json", issues) if attempt and (attempt / "task.json").is_file() else {}
        logical_seed = marker.get("training_seed", marker.get("seed", "")) if isinstance(marker, dict) else ""
        if isinstance(task_json, dict):
            logical_seed = task_json.get("seed", task_json.get("training_seed", logical_seed))
        rows.append({
            "analysis_group": label, "logical_task_root": str(task_root), "marker_path": str(marker_path),
            "selected_attempt": str(attempt) if attempt else "", "selected_attempt_exists": bool(attempt and attempt.is_dir()),
            "attempt_count": len(attempts), "result_file_count": len(prediction_files) + len(result_files),
            "prediction_file_count": len(prediction_files), "history_file_count": len(result_files),
            "selected_test_rmse": rmse, "selected_test_rmse_source": rmse_source,
            "logical_seed": logical_seed, "config_fingerprint": marker.get("config_fingerprint", "") if isinstance(marker, dict) else "",
            "task_identity": json.dumps(task_json.get("spec", task_json.get("parameters", {})), ensure_ascii=False, sort_keys=True) if isinstance(task_json, dict) else "",
            "parameter_values": json.dumps(resolved, ensure_ascii=False, sort_keys=True),
            "parameter_sources": json.dumps(sources, ensure_ascii=False, sort_keys=True),
            "lineage_evidence": ";".join(lineage), "symlink_task_root": task_root.is_symlink(),
            "symlink_selected_attempt": bool(attempt and attempt.is_symlink()),
            "success_final_adopted": valid and rmse not in {"", "conflict"},
            "status": "selected_attempt_valid" if valid and rmse not in {"", "conflict"} else "needs_review",
        })
    return rows


def expected_status(values: dict[str, Any]) -> tuple[str, list[str]]:
    detail = []
    for name, expected in EXPECTED.items():
        value = values.get(name, "")
        if value == "conflict":
            detail.append(name + "=conflict")
        elif value == "":
            detail.append(name + "=missing")
        else:
            try:
                ok = float(value) == float(expected)
            except (TypeError, ValueError):
                ok = str(value) == str(expected)
            detail.append(f"{name}={'match' if ok else 'mismatch:' + str(value)}")
    return ("confirmed" if all(item.endswith("=match") for item in detail) else "not_confirmed"), detail


def alignment(parameter_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], prior_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    def aggregate(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        configurations = [json.loads(row["parameter_values"]) for row in rows if row["parameter_values"]]
        checks = [expected_status(config)[0] for config in configurations]
        success = [row for row in rows if str(row["success_final_adopted"]) == "True"]
        return {"role": label, "logical_tasks": len(rows), "selected_attempts": len(success),
                "attempts_total": sum(int(row["attempt_count"]) for row in rows),
                "result_files": sum(int(row["result_file_count"]) for row in rows),
                "duplicate_or_retry_attempts": sum(max(0, int(row["attempt_count"]) - 1) for row in rows),
                "expected_parameter_status": "confirmed" if rows and all(check == "confirmed" for check in checks) else "not_confirmed",
                "evidence_paths": ";".join(row["logical_task_root"] for row in rows)}
    output = [aggregate("parameter_selection_extra_tau16", parameter_rows), aggregate("final_ablation_full", full_rows)]
    main_candidates = [row for row in prior_rows if row.get("registered_model") in {"hingemix", "ggpl_gtm"}]
    output.append({"role": "formal_main_result_candidates", "logical_tasks": len({row.get("run_dir") for row in main_candidates}),
                   "selected_attempts": "", "attempts_total": "", "result_files": len(main_candidates),
                   "duplicate_or_retry_attempts": "", "expected_parameter_status": "not_inferred_from_name",
                   "evidence_paths": ";".join(row.get("run_dir", "") for row in main_candidates)})
    pairs = {}
    for row in parameter_rows:
        pairs[(Path(row["logical_task_root"]).parent.name, str(row["logical_seed"]))] = row
    matched, identical, mismatch = 0, 0, 0
    for row in full_rows:
        key = (Path(row["logical_task_root"]).parent.name, str(row["logical_seed"]))
        other = pairs.get(key)
        if other and row["selected_test_rmse"] not in {"", "conflict"} and other["selected_test_rmse"] not in {"", "conflict"}:
            matched += 1
            if row["selected_test_rmse"] == other["selected_test_rmse"]:
                identical += 1
            else:
                mismatch += 1
    output.append({"role": "extra_tau16_to_full_pairing", "logical_tasks": matched, "selected_attempts": identical,
                   "attempts_total": mismatch, "result_files": "", "duplicate_or_retry_attempts": "",
                   "expected_parameter_status": "identical_test_rmse_pairs" if matched and identical == matched else "not_all_identical_or_unpaired",
                   "evidence_paths": "paired by dataset and seed from selected attempt markers; equal metrics alone do not establish reuse"})
    return output


def parse_feature_names(info: dict[str, Any]) -> list[str]:
    def visit(value: Any) -> list[str] | None:
        if not isinstance(value, dict):
            return None
        for key, child in value.items():
            if key.lower() in {"num_feature_names", "numerical_feature_names", "numeric_feature_names", "feature_names"}:
                if isinstance(child, list) and all(isinstance(item, str) for item in child):
                    return child
            found = visit(child)
            if found is not None:
                return found
        return None
    found = visit(info)
    if found is not None:
        return found
    return []


def metric_summary(y: Any, reconstructed: Any) -> dict[str, Any]:
    valid = np.isfinite(y) & np.isfinite(reconstructed)
    error = reconstructed[valid] - y[valid]
    nonfinite = int((~np.isfinite(reconstructed)).sum())
    return {"valid_samples": int(valid.sum()), "nonfinite_reconstructed": nonfinite,
            "mae": float(np.abs(error).mean()) if error.size else "", "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else "",
            "max_abs_error": float(np.abs(error).max()) if error.size else "",
            "label_min": float(np.nanmin(y)) if y.size else "", "label_max": float(np.nanmax(y)) if y.size else ""}


def generation_hits(project: Path, generator_root: Path | None, dataset: str) -> list[str]:
    """Search source trees only; never search the host filesystem or results."""
    roots = [project / "data", project / "scripts"]
    if generator_root is not None and generator_root.is_dir():
        roots.append(generator_root)
    hits = []
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in ("*.py", "*.sh", "*.md"):
            for path in root.rglob(suffix):
                try:
                    if path.stat().st_size > 2 * 1024 * 1024:
                        continue
                    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        text = line.lower()
                        if dataset in text or ("ipc" in text and ("instruction" in text or "cycle" in text)):
                            hits.append(f"{path}:{line_number}")
                            break
                except OSError:
                    continue
    return hits


def ipc_rows(project: Path, generator_root: Path | None, data_root: Path, issues: list[dict[str, str]], unresolved: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        directory = dataset_dir(data_root, dataset)
        if directory is None:
            rows.append({"dataset": dataset, "split": "", "status": "dataset_not_available_at_current_root", "evidence": str(data_root)})
            continue
        info = read_json(directory / "info.json", issues) or {}
        source_hits = generation_hits(project, generator_root, dataset)
        names = parse_feature_names(info if isinstance(info, dict) else {})
        label_name = ""
        if isinstance(info, dict):
            for key in ("label_name", "target_name", "target", "y_name"):
                if key in info:
                    label_name = str(info[key])
                    break
        normalized = [name.strip().lower() for name in names]
        inst = [index for index, name in enumerate(normalized) if name in IPC_INSTRUCTIONS]
        cycles = [index for index, name in enumerate(normalized) if name in IPC_CYCLES]
        related = [name for name in names if any(token in name.lower() for token in ("inst", "cycle", "ipc"))]
        can_compute = len(inst) == 1 and len(cycles) == 1
        for split in ("train", "val", "test"):
            row = {"dataset": dataset, "split": split, "label_name": label_name, "label_formula_status": "not_found_in_info_or_located_generator_source",
                   "target_scale_status": "stored_y_array_scale_not_proven_without_generation_or_run_transform_evidence", "feature_name_evidence": str(directory / "info.json"),
                   "instructions_candidates": ";".join(names[i] for i in inst), "cycles_candidates": ";".join(names[i] for i in cycles),
                   "related_event_columns": ";".join(related), "array_evidence": "", "status": "not_reconstructed", "interpretation": ""}
            row["generation_source_candidates"] = ";".join(source_hits)
            x_path, y_path = directory / f"X_num_{split}.npy", directory / f"y_{split}.npy"
            if not can_compute:
                row.update(status="unambiguous_total_instructions_and_cycles_not_found", interpretation="Column names are candidates only; migrations and stall-cycle fields are intentionally not treated as totals.")
            elif not x_path.is_file() or not y_path.is_file() or np is None:
                row.update(status="arrays_unavailable", interpretation="Cannot compute candidate ratio without source arrays and NumPy.")
            else:
                _, _, x = array_info(x_path, issues)
                _, _, y = array_info(y_path, issues)
                x, y = np.asarray(x), np.asarray(y).reshape(-1)
                denominator, numerator = x[:, cycles[0]], x[:, inst[0]]
                zero = int((denominator == 0).sum())
                with np.errstate(divide="ignore", invalid="ignore"):
                    reconstructed = numerator / denominator
                row.update(metric_summary(y, reconstructed), zero_denominator=zero,
                           array_evidence=f"{x_path};{y_path}", status="candidate_ratio_computed",
                           interpretation="Computed from stored dataset arrays only. It does not prove arrays are raw counters or that label windows/units/aggregation match.")
            rows.append(row)
        unresolved.append({"scope": dataset, "item": "IPC label provenance", "reason": "Need source generation/cleaning script and counter-window/unit definitions to interpret any candidate ratio.", "evidence": str(directory / "info.json") + ";" + ";".join(source_hits), "next_step": "Provide referenced generator or raw CSV provenance; do not infer leakage from column names or correlation."})
    return rows


def baseline_protocol(prior_rows: list[dict[str, str]], issues: list[dict[str, str]], unresolved: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for model, display in BASELINES.items():
        for dataset in DATASETS:
            candidates = [row for row in prior_rows if row.get("registered_model") == model and row.get("dataset") == dataset]
            inspected = []
            for row in candidates:
                run = Path(row["run_dir"])
                configs = configuration_sources(run, run, issues)
                values, sources, _ = parameter_evidence(configs)
                logs = [str(path) for path in (run / "training.log", run / "train.log", run / "command.txt") if path.is_file()]
                inspected.append({"path": str(run), "seed": row.get("training_seed", "unknown"), "config_sources": sources, "logs": logs,
                                  "data_config": str(run / "data_config.yaml") if (run / "data_config.yaml").is_file() else ""})
            seeds = sorted({item["seed"] for item in inspected if item["seed"].isdigit()}, key=int)
            unknown = sum(item["seed"] in {"", "unknown", "conflict"} for item in inspected)
            status = "records_found_evidence_incomplete" if inspected else "not_located_in_first_audit_records"
            evidence = json.dumps(inspected, ensure_ascii=False)
            rows.append({"model": display, "registered_model": model, "dataset": dataset, "run_directories": len(inspected),
                         "recovered_training_seeds": ";".join(seeds), "unknown_or_conflicting_seed_records": unknown,
                         "data_config_paths": ";".join(item["data_config"] for item in inspected if item["data_config"]),
                         "configuration_and_log_evidence": evidence, "status": status,
                         "rmse_recalculation": "not_attempted_without_saved_target_array_and_verified_target_scale"})
            if inspected:
                unresolved.append({"scope": f"{display}/{dataset}", "item": "baseline protocol", "reason": "Need recoverable training seed, split/index evidence, target-scale evidence and actual training configuration before paper reuse.", "evidence": ";".join(item["path"] for item in inspected), "next_step": "Inspect listed task/log/config artifacts; recompute RMSE only if saved predictions and aligned raw targets are available."})
    return rows


def run_audit(project: Path, previous: Path, results: Path, data: Path, generator_root: Path | None, output: Path) -> None:
    issues: list[dict[str, str]] = []
    prior = csv_rows(previous / "experiment_records.csv")
    parameter_root = results / "ggpl_gtm_parameter_supplement" / "extras" / "extra_tau16"
    full_root = results / "ggpl_gtm_ablation_final_tau16" / "full"
    parameter_rows = lineage_for_tree("extra_tau16", parameter_root, issues)
    full_rows = lineage_for_tree("ablation_full", full_root, issues)
    unresolved: list[dict[str, str]] = []
    ipc = ipc_rows(project, generator_root, data, issues, unresolved)
    baseline = baseline_protocol(prior, issues, unresolved)
    align = alignment(parameter_rows, full_rows, prior)
    corrections = [
        {"correction": "configuration_evidence", "status": "implemented", "detail": "Reads task.json, parameter_result.json, parameter_task.json and final_config.yaml; preserves field-level source/conflict evidence."},
        {"correction": "run_counting", "status": "implemented", "detail": "Counts task-root parameter_result markers as logical tasks; selected attempts, retries and JSON result files are separate fields."},
        {"correction": "formal_main_lookup", "status": "implemented", "detail": "Uses previous audit records with registered_model hingemix/ggpl_gtm as candidates; does not require results/hingemix."},
    ]
    actions: list[dict[str, str]] = []
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "audit_corrections.csv", corrections, ["correction", "status", "detail"])
    write_csv(output / "run_lineage.csv", parameter_rows + full_rows, list((parameter_rows + full_rows)[0]) if parameter_rows or full_rows else ["analysis_group", "logical_task_root", "selected_attempt"])
    write_csv(output / "final_model_alignment.csv", align, list(align[0]))
    write_csv(output / "ipc_reconstruction_checks.csv", ipc, sorted({key for row in ipc for key in row}))
    write_csv(output / "baseline_protocol_audit.csv", baseline, list(baseline[0]))
    write_csv(output / "unresolved_items.csv", unresolved, ["scope", "item", "reason", "evidence", "next_step"])
    write_csv(output / "confirmed_actions.csv", actions, ["scope", "action", "evidence", "reason"])
    write_csv(output / "scan_issues.csv", issues, ["kind", "path", "detail"])
    matched = next((row for row in align if row["role"] == "extra_tau16_to_full_pairing"), {})
    summary = ["# HingeMix targeted audit v2", "", f"Generated: {now()}", f"Previous audit: `{previous}`", f"Results root: `{results}`", f"Data root: `{data}`", "",
               "## Scope", "This run reads only the prior audit record paths and the two fixed HingeMix trees. It does not recursively rescan the entire results directory, train models, or edit existing experiment artifacts.", "",
               "## Counts", f"extra_tau16 logical task markers: {len(parameter_rows)}; full logical task markers: {len(full_rows)}.",
               f"Paired dataset/seed entries: {matched.get('logical_tasks', 0)}; identical selected test-RMSE pairs: {matched.get('selected_attempts', 0)}; mismatches: {matched.get('attempts_total', 0)}.",
               "Equal metrics are recorded as numerical identity only. Reuse is confirmed only when explicit lineage fields, symlinks, or scheduler/manifest mapping evidence is present.", "",
               "## Limits", "IPC candidate ratios are interpreted only after source-array scale, sampling window, units, aggregation and label-generation evidence are established. Missing historical seed/configuration evidence remains unresolved rather than becoming a rerun request."]
    (output / "audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote targeted read-only audit to {output}")
    print(f"extra_tau16_tasks={len(parameter_rows)} full_tasks={len(full_rows)} baseline_rows={len(baseline)} issues={len(issues)}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prior = root / "previous"; prior.mkdir(parents=True)
        write_csv(prior / "experiment_records.csv", [{"registered_model": "lightgbm", "dataset": "hpcg2", "run_dir": str(root / "results" / "lightgbm" / "hpcg2"), "training_seed": "unknown"}], ["registered_model", "dataset", "run_dir", "training_seed"])
        for base in (root / "results" / "ggpl_gtm_parameter_supplement" / "extras" / "extra_tau16", root / "results" / "ggpl_gtm_ablation_final_tau16" / "full"):
            attempt = base / "hpcg2" / "seed_0" / "attempts" / "attempt_001"; attempt.mkdir(parents=True)
            config = {"model": {key: value for key, value in EXPECTED.items() if key not in {"lr", "batch_size"}}, "training": {"lr": 1e-5, "batch_size": 32}}
            (attempt / "final_config.yaml").write_text("model:\n  num_breakpoints: 8\n  graph_dynamic_rank: 16\n  graph_temperature: 16.0\n  d_token: 1024\n  n_layers: 1\ntraining:\n  lr: 1.0e-5\n  batch_size: 32\n", encoding="utf-8")
            (attempt / "task.json").write_text(json.dumps({"seed": 0, "config_fingerprint": "x"}), encoding="utf-8")
            (attempt / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.0, "metrics": {"rmse": 1.0}}), encoding="utf-8")
            (attempt.parents[1] / "parameter_result.json").write_text(json.dumps({"attempt_dir": str(attempt), "config_fingerprint": "x"}), encoding="utf-8")
        output = root / "out"
        run_audit(root, prior, root / "results", root / "data", None, output)
        rows = list(csv.DictReader((output / "run_lineage.csv").open(encoding="utf-8-sig")))
        assert len(rows) == 2 and all(row["attempt_count"] == "1" for row in rows)
        paired = list(csv.DictReader((output / "final_model_alignment.csv").open(encoding="utf-8-sig")))
        assert next(row for row in paired if row["role"] == "extra_tau16_to_full_pairing")["selected_attempts"] == "1"
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--previous-audit", type=Path, help="First-audit directory containing experiment_records.csv.")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--generator-root", type=Path, help="Optional known generator project; no host-wide search is performed.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.previous_audit is None:
        parser.error("--previous-audit is required unless --self-test is used")
    project = args.project_root.resolve()
    output = args.output or project / "audit" / ("hingemix_data_records_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    generator = args.generator_root.resolve() if args.generator_root else project.parent / "server-modeling-and-evaluation"
    run_audit(project, args.previous_audit.resolve(), (args.results_root or project / "results").resolve(), (args.data_root or project / "data").resolve(), generator, output.resolve())


if __name__ == "__main__":
    main()
