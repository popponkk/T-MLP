#!/usr/bin/env python3
"""Read-only audit for HingeMix data provenance and historical experiment records.

The script never imports the training entry point, writes only below --output, and
does not load checkpoints.  It is intended to be run on the server that contains
the actual datasets and results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # The server project normally has NumPy; keep failure explicit.
    np = None


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
DISPLAY_DATASET = {"mix_with_five_datasets161": "mix"}
BASELINES = {
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
    "excel-former": "ExcelFormer", "ft-transformer": "FT-Transformer",
    "node": "NODE", "mlp": "MLP", "autoint": "AutoInt", "tabm": "TabM",
}
HINGEMIX_ALIASES = {"hingemix", "ggpl_gtm"}
FORMAL_MODELS = tuple(BASELINES) + ("hingemix",)
EXPECTED = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
    "lr": 1e-5,
    "batch_size": 32,
}
SEED_RE = re.compile(r"^seed[_-]?(\d+)$", re.I)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return ""


def read_json(path: Path, issues: list[dict[str, str]]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append({"kind": "json_read_error", "path": str(path), "detail": str(exc)})
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_json(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten_json(child, f"{prefix}[{index}]"))
    else:
        result[prefix] = value
    return result


def find_key_values(value: Any, names: set[str]) -> list[tuple[str, Any]]:
    return [(path, item) for path, item in flatten_json(value).items()
            if path.rsplit(".", 1)[-1].lower() in names]


def load_yaml_scalars(path: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    """Use PyYAML if installed; otherwise collect unambiguous top-level scalars."""
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        result: dict[str, Any] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                match = re.match(r"^([A-Za-z_][\w-]*):\s*([^#]+?)\s*$", line)
                if not match:
                    continue
                key, raw = match.groups()
                raw = raw.strip().strip("'\"")
                if raw.lower() in {"true", "false"}:
                    result[key] = raw.lower() == "true"
                else:
                    try:
                        result[key] = int(raw)
                    except ValueError:
                        try:
                            result[key] = float(raw)
                        except ValueError:
                            result[key] = raw
            return result
        except OSError as exc:
            issues.append({"kind": "yaml_read_error", "path": str(path), "detail": str(exc)})
            return {}
    except Exception as exc:
        issues.append({"kind": "yaml_parse_error", "path": str(path), "detail": str(exc)})
        return {}


def file_candidates(directory: Path, names: tuple[str, ...]) -> list[Path]:
    return [directory / name for name in names if (directory / name).is_file()]


def dataset_dir(data_root: Path, dataset: str) -> Path | None:
    candidates = (data_root / "datasets" / dataset, data_root / "custom_datasets" / dataset)
    return next((path for path in candidates if path.is_dir()), None)


def array_info(path: Path, issues: list[dict[str, str]]) -> tuple[int, tuple[int, ...], Any]:
    if np is None:
        return 0, (), None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(array.shape[0]) if array.ndim else 1, tuple(array.shape), array
    except Exception as exc:
        issues.append({"kind": "array_read_error", "path": str(path), "detail": str(exc)})
        return 0, (), None


def names_from_info(info: dict[str, Any], key_hints: tuple[str, ...]) -> list[str]:
    def visit(value: Any) -> list[str] | None:
        if not isinstance(value, dict):
            return None
        for key, child in value.items():
            if key.lower() in key_hints and isinstance(child, list) and all(isinstance(item, str) for item in child):
                return child
            nested = visit(child)
            if nested is not None:
                return nested
        return None

    found = visit(info)
    if found is not None:
        return found
    return []


def find_label(info: dict[str, Any]) -> tuple[str, str]:
    for path, value in flatten_json(info).items():
        if path.rsplit(".", 1)[-1].lower() in {"label_name", "target_name", "target", "y_name"}:
            return scalar(value), path
    return "", ""


def duplicate_summary(arrays: list[Any]) -> str:
    """Cheap exact-row duplicate check; no claim about semantic scenario duplicates."""
    if np is None or not arrays or any(array is None for array in arrays):
        return "not_checked"
    try:
        rows = np.concatenate([np.ascontiguousarray(array).reshape(array.shape[0], -1) for array in arrays])
        if rows.dtype == object:
            return "not_checked_object_dtype"
        row_bytes = rows.view(np.uint8).reshape(rows.shape[0], -1)
        unique = np.unique(row_bytes, axis=0).shape[0]
        return f"exact_duplicates={rows.shape[0] - unique} across_loaded_splits"
    except Exception:
        return "not_checked"


def audit_dataset(data_root: Path, dataset: str, issues: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = dataset_dir(data_root, dataset)
    base = {
        "dataset": dataset, "display_dataset": DISPLAY_DATASET.get(dataset, dataset),
        "dataset_dir": str(directory) if directory else "", "dataset_dir_exists": bool(directory),
        "data_files": "", "data_hashes": "", "raw_sample_count": "", "used_sample_count": "",
        "numeric_feature_count": "", "categorical_feature_count": "", "feature_names_source": "",
        "label_name": "", "label_evidence": "", "split_counts": "", "split_evidence": "",
        "duplicate_check": "", "preprocessing_evidence": "", "breakpoint_cache_evidence": "",
        "status": "current_check_location_not_found" if not directory else "evidence_collected",
        "notes": "No local dataset directory; this is not evidence about the server." if not directory else "",
    }
    target = {
        "dataset": dataset, "label_name": "", "label_formula": "not_found_in_dataset_metadata",
        "formula_evidence": "", "input_columns": "", "ipc_columns": "", "instructions_columns": "",
        "cycles_columns": "", "related_rate_or_ratio_columns": "", "reconstruction_status": "not_attempted",
        "reconstruction_error": "", "leakage_assessment": "insufficient_evidence", "evidence": "",
    }
    if not directory:
        return base, target

    info_path = directory / "info.json"
    info = read_json(info_path, issues) if info_path.is_file() else {}
    if not isinstance(info, dict):
        info = {}
    numeric_names = names_from_info(info, ("num_feature_names", "numerical_feature_names", "numeric_feature_names"))
    cat_names = names_from_info(info, ("cat_feature_names", "categorical_feature_names"))
    label, label_path = find_label(info)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    base["data_files"] = ";".join(path.name for path in files)
    base["data_hashes"] = ";".join(f"{path.name}:{sha256(path)}" for path in files)
    base["label_name"], base["label_evidence"] = label, f"{info_path}:{label_path}" if label_path else ""
    target["label_name"] = label
    target["input_columns"] = ";".join(numeric_names + cat_names)
    target["evidence"] = base["label_evidence"]
    base["feature_names_source"] = str(info_path) if numeric_names or cat_names else "not_found"

    split_counts, num_shapes, cat_shapes, y_arrays, numeric_arrays = {}, [], [], [], []
    for split in ("train", "val", "test"):
        for prefix, shapes, arrays in (("X_num", num_shapes, numeric_arrays), ("X_cat", cat_shapes, []), ("y", [], y_arrays)):
            path = directory / f"{prefix}_{split}.npy"
            if path.is_file():
                count, shape, array = array_info(path, issues)
                split_counts.setdefault(split, count)
                shapes.append(shape)
                if prefix == "X_num":
                    numeric_arrays.append(array)
    base["split_counts"] = json.dumps(split_counts, sort_keys=True)
    base["split_evidence"] = str(directory / "idx_train.npy") if (directory / "idx_train.npy").is_file() else "array_row_counts"
    base["used_sample_count"] = sum(split_counts.values()) if split_counts else ""
    base["raw_sample_count"] = info.get("n_samples", info.get("sample_count", ""))
    if num_shapes:
        base["numeric_feature_count"] = num_shapes[0][1] if len(num_shapes[0]) > 1 else 0
    if cat_shapes:
        base["categorical_feature_count"] = cat_shapes[0][1] if len(cat_shapes[0]) > 1 else 0
    base["duplicate_check"] = duplicate_summary(numeric_arrays)
    base["preprocessing_evidence"] = "data/processor.py: Processor.apply fits transformations during current loading; historical fit scope requires per-run evidence"
    cache_dirs = list(directory.glob("cache__*.pickle"))
    base["breakpoint_cache_evidence"] = "feature cache files=" + str(len(cache_dirs)) + "; GGPL breakpoint cache must be checked per result directory"

    lower_columns = [name.lower() for name in numeric_names + cat_names]
    def selected(predicate):
        return [name for name in numeric_names + cat_names if predicate(name.lower())]
    ipc, instructions, cycles = selected(lambda name: "ipc" in name), selected(lambda name: "instruction" in name), selected(lambda name: "cycle" in name)
    related = selected(lambda name: any(token in name for token in ("rate", "ratio", "per_", "_per", "throughput")))
    target.update({"ipc_columns": ";".join(ipc), "instructions_columns": ";".join(instructions), "cycles_columns": ";".join(cycles), "related_rate_or_ratio_columns": ";".join(related)})
    if ipc or (instructions and cycles):
        target["reconstruction_status"] = "related_fields_present_but_window_and_aggregation_not_confirmed"
        target["leakage_assessment"] = "related_fields_require_manual_range_unit_and_aggregation_check"
    else:
        target["reconstruction_status"] = "no_direct_reconstruction_path_found_in_named_columns"
        target["leakage_assessment"] = "not_confirmed_safe_without_raw_generation_metadata"
    return base, target


def path_parts(path: Path, root: Path) -> list[str]:
    try:
        return [part.lower() for part in path.relative_to(root).parts]
    except ValueError:
        return [part.lower() for part in path.parts]


def find_path_model(parts: list[str]) -> str | None:
    exact = set(BASELINES) | HINGEMIX_ALIASES
    hits = [part for part in parts if part in exact]
    return hits[-1] if len(set(hits)) == 1 else None


def find_path_dataset(parts: list[str]) -> str | None:
    hits = [part for part in parts if part in DATASETS]
    return hits[-1] if len(set(hits)) == 1 else None


def find_seed_from_path(parts: list[str]) -> int | None:
    for part in reversed(parts):
        match = SEED_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def nearby_metadata(run_dir: Path, root: Path, issues: list[dict[str, str]]) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    directory = run_dir
    for _ in range(6):
        for name in ("parameter_result.json", "parameter_task.json", "task.json", "run.json", "config.json"):
            path = directory / name
            if path.is_file():
                value = read_json(path, issues)
                if isinstance(value, dict):
                    result.append((path, value))
        if directory == root or directory.parent == directory:
            break
        directory = directory.parent
    return result


def metadata_identity(metadata: list[tuple[Path, dict[str, Any]]]) -> tuple[set[str], set[str]]:
    models, datasets = set(), set()
    for _, content in metadata:
        for _, value in find_key_values(content, {"model", "model_name", "dataset", "dataset_name"}):
            if not isinstance(value, str):
                continue
            value = value.lower()
            if value in BASELINES or value in HINGEMIX_ALIASES:
                models.add(value)
            if value in DATASETS:
                datasets.add(value)
    return models, datasets


def extract_seed(metadata: list[tuple[Path, dict[str, Any]]], path_seed: int | None) -> tuple[str, str]:
    observed = []
    if path_seed is not None:
        observed.append(("path", path_seed))
    for path, content in metadata:
        for field, value in find_key_values(content, {"training_seed", "seed", "random_seed"}):
            if isinstance(value, int) and not isinstance(value, bool):
                observed.append((f"{path}:{field}", value))
    values = {value for _, value in observed}
    if not values:
        return "unknown", ""
    if len(values) > 1:
        return "conflict", ";".join(f"{source}={value}" for source, value in observed)
    return str(next(iter(values))), ";".join(f"{source}={value}" for source, value in observed)


def extract_value(metadata: list[tuple[Path, dict[str, Any]]], names: set[str]) -> tuple[str, str]:
    found = []
    for path, content in metadata:
        for field, value in find_key_values(content, names):
            found.append((f"{path}:{field}", value))
    unique = {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for _, value in found}
    if len(unique) != 1:
        return ("conflict" if len(unique) > 1 else ""), ";".join(f"{source}={scalar(value)}" for source, value in found)
    return scalar(found[0][1]), found[0][0]


def extract_metrics(prediction: dict[str, Any] | None, history: dict[str, Any] | None) -> tuple[str, str, str, str]:
    test, test_source, validation, validation_source = "", "", "", ""
    if isinstance(prediction, dict):
        if prediction.get("metric_name") == "rmse" and finite(prediction.get("metric")):
            test, test_source = str(prediction["metric"]), "prediction.json:metric (metric_name=rmse)"
        metrics = prediction.get("metrics")
        if isinstance(metrics, dict) and finite(metrics.get("rmse")):
            candidate = str(metrics["rmse"])
            source = "prediction.json:metrics.rmse"
            if test and test != candidate:
                test, test_source = "conflict", test_source + ";" + source
            elif not test:
                test, test_source = candidate, source
    if isinstance(history, dict):
        val = history.get("val")
        if isinstance(val, dict) and val.get("metric_name") == "rmse" and finite(val.get("best_metric")):
            validation, validation_source = str(val["best_metric"]), "results.json:val.best_metric"
    return test, test_source, validation, validation_source


def scan_results(results_root: Path, issues: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    if not results_root.is_dir():
        return [], 0
    records, seen, scanned = [], set(), 0
    for current, dirs, files in os.walk(results_root, followlinks=False):
        scanned += len(files)
        if "prediction.json" not in files and "results.json" not in files:
            continue
        run_dir = Path(current)
        resolved = str(run_dir.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        parts = path_parts(run_dir, results_root)
        metadata = nearby_metadata(run_dir, results_root, issues)
        path_model, path_dataset = find_path_model(parts), find_path_dataset(parts)
        meta_models, meta_datasets = metadata_identity(metadata)
        model = path_model or (next(iter(meta_models)) if len(meta_models) == 1 else "")
        dataset = path_dataset or (next(iter(meta_datasets)) if len(meta_datasets) == 1 else "")
        conflict = ""
        if path_model and meta_models and path_model not in meta_models:
            conflict += "model_path_metadata_conflict;"
        if path_dataset and meta_datasets and path_dataset not in meta_datasets:
            conflict += "dataset_path_metadata_conflict;"
        prediction = read_json(run_dir / "prediction.json", issues) if (run_dir / "prediction.json").is_file() else None
        history = read_json(run_dir / "results.json", issues) if (run_dir / "results.json").is_file() else None
        test, test_source, validation, validation_source = extract_metrics(prediction, history)
        seed, seed_evidence = extract_seed(metadata, find_seed_from_path(parts))
        data_seed, data_seed_evidence = extract_value(metadata, {"data_seed", "split_seed"})
        gbdt_seed, gbdt_seed_evidence = extract_value(metadata, {"gbdt_seed", "breakpoint_seed"})
        config_values = {}
        config_sources = []
        for directory in (run_dir, *[path.parent for path, _ in metadata]):
            for name in ("parameter_config.yaml", "data_config.yaml", "config.yaml"):
                path = directory / name
                if path.is_file():
                    config_values.update(load_yaml_scalars(path, issues))
                    config_sources.append(str(path))
        for _, content in metadata:
            config_values.update({key: value for key, value in flatten_json(content).items() if key.rsplit(".", 1)[-1] in EXPECTED})
        record = {
            "model_identity": "HingeMix" if model in HINGEMIX_ALIASES else BASELINES.get(model, model or "unknown"),
            "registered_model": model, "dataset": dataset, "display_dataset": DISPLAY_DATASET.get(dataset, dataset),
            "training_seed": seed, "training_seed_evidence": seed_evidence,
            "data_seed": data_seed, "data_seed_evidence": data_seed_evidence,
            "gbdt_seed": gbdt_seed, "gbdt_seed_evidence": gbdt_seed_evidence,
            "test_rmse": test, "test_rmse_source": test_source,
            "validation_rmse": validation, "validation_rmse_source": validation_source,
            "run_dir": str(run_dir), "prediction_path": str(run_dir / "prediction.json") if prediction is not None else "",
            "results_path": str(run_dir / "results.json") if history is not None else "",
            "metadata_paths": ";".join(str(path) for path, _ in metadata),
            "config_paths": ";".join(dict.fromkeys(config_sources)),
            "config_values": json.dumps(config_values, ensure_ascii=False, sort_keys=True),
            "identity_conflict": conflict.rstrip(";"),
            "status": "found_but_evidence_insufficient",
        }
        if conflict or seed == "conflict" or test == "conflict":
            record["status"] = "evidence_conflict"
        elif model and dataset and test and seed != "unknown":
            record["status"] = "found_requires_protocol_check"
        records.append(record)
    return records, scanned


def baseline_rows(records: list[dict[str, Any]], unresolved: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for model, display in BASELINES.items():
        for dataset in DATASETS:
            matches = [record for record in records if record["registered_model"] == model and record["dataset"] == dataset]
            known = sorted({record["training_seed"] for record in matches if record["training_seed"].isdigit()})
            unknown = sum(record["training_seed"] == "unknown" for record in matches)
            conflicts = sum(record["status"] == "evidence_conflict" for record in matches)
            status = "not_located_at_current_check_root" if not matches else "found_but_protocol_not_confirmed"
            reason = "No matching record was located; this is not proof the server never ran it." if not matches else "Historical result found; dataset version, split, target scale and actual configuration require evidence."
            if conflicts:
                status, reason = "evidence_conflict", "At least one matching record has conflicting identity, seed, or metric evidence."
            rows.append({"model": display, "registered_model": model, "dataset": dataset,
                         "valid_seeds_confirmed": ";".join(known), "unknown_seed_records": unknown,
                         "record_count": len(matches), "status": status, "reason": reason,
                         "record_paths": ";".join(record["run_dir"] for record in matches)})
            if matches:
                unresolved.append({"scope": f"{display}/{dataset}", "item": "protocol comparability",
                                   "reason": "Need historical data version, split, target-scale and configuration evidence before reuse.",
                                   "search_locations": ";".join(record["run_dir"] for record in matches), "next_step": "Inspect referenced config, data_config and logs; do not retrain yet."})
    return rows


def expected_match(values: dict[str, Any]) -> tuple[bool, str]:
    if not values:
        return False, "no_actual_configuration_snapshot"
    flat = flatten_json(values)
    observed = {}
    for key, expected in EXPECTED.items():
        candidates = [value for path, value in flat.items() if path.rsplit(".", 1)[-1] == key]
        if not candidates:
            observed[key] = "missing"
        elif any(str(value) == str(expected) for value in candidates):
            observed[key] = "match"
        else:
            observed[key] = "mismatch:" + ",".join(map(str, candidates))
    return all(value == "match" for value in observed.values()), json.dumps(observed, sort_keys=True)


def alignment_rows(project_root: Path, results_root: Path, records: list[dict[str, Any]], issues: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = {
        "parameter_selection_extra_tau16": results_root / "ggpl_gtm_parameter_supplement" / "extras" / "extra_tau16",
        "final_ablation_full": results_root / "ggpl_gtm_ablation_final_tau16" / "full",
        "formal_main_result": results_root / "hingemix",
    }
    rows = []
    for role, path in targets.items():
        snapshots = []
        if path.exists():
            for config in path.rglob("parameter_config.yaml"):
                snapshots.append(load_yaml_scalars(config, issues))
        matched, detail = expected_match(snapshots[0] if snapshots else {})
        rows.append({"role": role, "expected_identity": "single_projection_hingemix",
                     "expected_parameters": json.dumps(EXPECTED, sort_keys=True), "evidence_path": str(path),
                     "path_exists": path.exists(), "config_snapshot_count": len(snapshots),
                     "expected_config_match": matched, "config_check": detail,
                     "result_records_under_path": sum(record["run_dir"].startswith(str(path)) for record in records),
                     "alignment_status": "not_confirmed" if not matched else "parameter_match_only_split_and_code_identity_pending"})
    return rows


def summary_text(project_root: Path, results_root: Path, output: Path, datasets: list[dict[str, Any]], records: list[dict[str, Any]], scanned: int, issues: list[dict[str, str]]) -> str:
    found_data = sum(row["dataset_dir_exists"] for row in datasets)
    found_formal = sum(record["registered_model"] in BASELINES or record["registered_model"] in HINGEMIX_ALIASES for record in records)
    lines = [
        "# HingeMix final data and record audit",
        "", f"Generated: {now()}", f"Project root: `{project_root}`", f"Results root: `{results_root}`", f"Output: `{output}`", "",
        "## Scope and method",
        "This is a read-only audit. It does not run training, load checkpoints, modify data, or treat an absent local result directory as evidence that experiments were not run on another machine.",
        f"Dataset directories located: {found_data}/{len(DATASETS)}. Result files scanned: {scanned}. Formal-model result records parsed: {found_formal}. Parse/issues recorded: {len(issues)}.",
        "", "## Confirmed by static source inspection",
        "- Current data processing is entered through `main.py` -> `DataProcessor.load_preproc_default`; split persistence and transformations require per-run artifacts to establish historical equivalence.",
        "- RMSE implementation can multiply by target standard deviation (`utils/metrics.py`); historical raw-scale RMSE still requires the actual run's target-transform evidence.",
        "- HingeMix final identity/configuration, data split, and historical record equivalence are not inferred from directory names alone.",
        "", "## Interpretation",
        "- `found_but_protocol_not_confirmed` means a numerical record was found, not that it is paper-ready.",
        "- `not_located_at_current_check_root` means only that this scan root did not expose a matching record.",
        "- No retraining is recommended solely from missing evidence; use `unresolved_items.csv` to locate configurations, logs, manifests, and prediction artifacts first.",
    ]
    if not results_root.is_dir():
        lines += ["", "## Environment limitation", "The supplied results root does not exist in this environment. This report is a local capability check, not a judgment about the server experiments. Run this script on the server with its real `results` directory."]
    return "\n".join(lines) + "\n"


def run_audit(project_root: Path, results_root: Path, data_root: Path, output: Path) -> None:
    issues: list[dict[str, str]] = []
    datasets, targets = [], []
    for dataset in DATASETS:
        dataset_row, target_row = audit_dataset(data_root, dataset, issues)
        datasets.append(dataset_row)
        targets.append(target_row)
    records, scanned = scan_results(results_root, issues)
    unresolved: list[dict[str, str]] = []
    baseline = baseline_rows(records, unresolved)
    alignment = alignment_rows(project_root, results_root, records, issues)
    for row in datasets:
        if not row["dataset_dir_exists"]:
            unresolved.append({"scope": row["dataset"], "item": "dataset evidence", "reason": "Dataset directory unavailable at current data root.", "search_locations": str(data_root), "next_step": "Run on server or pass --data-root containing the real datasets."})
    confirmed_actions: list[dict[str, str]] = []
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "dataset_audit.csv", datasets, list(datasets[0]))
    write_csv(output / "feature_target_checks.csv", targets, list(targets[0]))
    record_fields = ["model_identity", "registered_model", "dataset", "display_dataset", "training_seed", "training_seed_evidence", "data_seed", "data_seed_evidence", "gbdt_seed", "gbdt_seed_evidence", "test_rmse", "test_rmse_source", "validation_rmse", "validation_rmse_source", "run_dir", "prediction_path", "results_path", "metadata_paths", "config_paths", "config_values", "identity_conflict", "status"]
    write_csv(output / "experiment_records.csv", records, record_fields)
    write_csv(output / "final_model_alignment.csv", alignment, list(alignment[0]))
    write_csv(output / "unresolved_items.csv", unresolved, ["scope", "item", "reason", "search_locations", "next_step"])
    write_csv(output / "confirmed_actions.csv", confirmed_actions, ["scope", "action", "evidence", "reason"])
    write_csv(output / "scan_issues.csv", issues, ["kind", "path", "detail"])
    (output / "audit_summary.md").write_text(summary_text(project_root, results_root, output, datasets, records, scanned, issues), encoding="utf-8")
    print(f"Wrote read-only audit to {output}")
    print(f"datasets_found={sum(row['dataset_dir_exists'] for row in datasets)}/{len(datasets)} records={len(records)} scanned_files={scanned} issues={len(issues)}")


def self_test() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for the audit self-test.")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        dataset = root / "data" / "custom_datasets" / "hpcg2"
        dataset.mkdir(parents=True)
        (dataset / "info.json").write_text(json.dumps({"num_feature_names": ["event_a", "cycles"], "label_name": "ipc"}), encoding="utf-8")
        for split in ("train", "val", "test"):
            np.save(dataset / f"X_num_{split}.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
            np.save(dataset / f"y_{split}.npy", np.array([1.0, 2.0]))
        run = root / "results" / "mlp" / "hpcg2" / "seed_3"
        run.mkdir(parents=True)
        (run / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.25}), encoding="utf-8")
        (run / "results.json").write_text(json.dumps({"val": {"metric_name": "rmse", "best_metric": 1.5}}), encoding="utf-8")
        false_match = root / "results" / "ggpl_tmlp" / "hpcg2"
        false_match.mkdir(parents=True)
        (false_match / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 9.0}), encoding="utf-8")
        output = root / "audit"
        run_audit(root, root / "results", root / "data", output)
        records = list(csv.DictReader((output / "experiment_records.csv").open(encoding="utf-8-sig")))
        assert len(records) == 2, "All records are retained, including unrecognized historical ones."
        mlp = next(row for row in records if row["registered_model"] == "mlp")
        assert mlp["training_seed"] == "3" and mlp["test_rmse"] == "1.25"
        assert mlp["validation_rmse"] == "1.5", "Validation must not be emitted as test RMSE."
        assert all(row["registered_model"] != "mlp" or "ggpl_tmlp" not in row["run_dir"] for row in records)
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    project_root = args.project_root.resolve()
    results_root = (args.results_root or project_root / "results").resolve()
    data_root = (args.data_root or project_root / "data").resolve()
    output = args.output or project_root / "audit" / ("hingemix_data_records_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_audit(project_root, results_root, data_root, output.resolve())


if __name__ == "__main__":
    main()
