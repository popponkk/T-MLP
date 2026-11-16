#!/usr/bin/env python3
"""Read-only inventory of formal-result artifacts usable for efficiency measurement."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_efficiency_common import (  # noqa: E402
    DATASETS, DISPLAY_DATASET, HINGEMIX_FINAL, MODELS, all_model_files, cache_candidates,
    checkpoint_path, config_model_name, configuration_path, data_config_path, finite, fingerprint,
    read_json, read_yaml, seed_from_path, selected_attempt, sha256_file, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="results/hingemix_efficiency")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--output", default=None, help="Explicit output directory; otherwise a timestamped audit directory is made")
    return parser.parse_args()


def model_config(document: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {}
    value = document.get("model")
    if not isinstance(value, dict):
        return document
    # HingeMix ablation-v2 stores the resolved constructor arguments here,
    # while the outer mapping records the ablation identity.  Merge both so
    # formal checks validate effective, rather than merely requested, values.
    effective = value.get("effective_model_config")
    return {**value, **effective} if isinstance(effective, dict) else value


def hinge_config_ok(config: dict[str, Any], training: dict[str, Any]) -> tuple[bool, str]:
    if config.get("ablation") not in ("full", None):
        return False, "not_full_ablation"
    for key, value in HINGEMIX_FINAL.items():
        current = config.get(key)
        if not finite(current) or float(current) != float(value):
            return False, f"formal_hingemix_{key}_mismatch"
    for key, value in {"lr": 1e-5, "batch_size": 32}.items():
        current = training.get(key)
        if not finite(current) or float(current) != float(value):
            return False, f"formal_hingemix_training_{key}_mismatch"
    return True, ""


def training_time_info(attempt: Path) -> tuple[str, str, str]:
    history = read_json(attempt / "results.json") or {}
    train = history.get("train") if isinstance(history.get("train"), dict) else {}
    value = train.get("tot_time")
    log = attempt / "training.log"
    if finite(value):
        return str(value), "model_fit_only_from_results.json; preprocessing/GBDT scope unresolved", "incomplete_scope"
    if log.is_file():
        return "", "training.log exists but no unambiguous whole-run wall time field", "untrusted"
    return "", "training log absent", "missing"


def make_record(model: str, dataset: str, task_root: Path, attempt: Path | None, association: str) -> dict[str, Any]:
    source = attempt or task_root
    config_path, data_path = configuration_path(source), data_config_path(source)
    config_doc = read_yaml(config_path) if config_path else None
    effective_name = config_model_name(config_path)
    config = model_config(config_doc)
    training = (config_doc or {}).get("training") if isinstance((config_doc or {}).get("training"), dict) else {}
    checkpoint = checkpoint_path(source, model)
    caches = cache_candidates(data_path)
    prediction = read_json(source / "prediction.json")
    saved_predictions = source / "test_predictions.npy"
    status, reason = "ready", ""
    if attempt is None:
        status, reason = "ambiguous_provenance", "selected attempt marker absent"
    elif config_path is None:
        status, reason = "ambiguous_provenance", "saved model configuration missing"
    elif effective_name is not None and effective_name not in {model, "hingemix_ablation_v2" if model == "hingemix" else model}:
        status, reason = "ambiguous_provenance", f"saved config model_name={effective_name!r}"
    elif model == "hingemix":
        ok, detail = hinge_config_ok(config, training)
        if not ok:
            status, reason = "configuration_mismatch", detail
    if status == "ready" and checkpoint is None:
        status, reason = "missing_checkpoint", "best.bin/final.bin or final.pkl missing"
    if status == "ready" and data_path is None:
        status, reason = "missing_preprocessor", "data_config.yaml missing"
    elif status == "ready" and not caches:
        status, reason = "missing_preprocessor", "no saved transformed Dataset cache found at data_config.cache_dir"
    if status == "ready" and not isinstance(prediction, dict):
        status, reason = "ambiguous_provenance", "final prediction.json missing"
    if status == "ready" and not saved_predictions.is_file():
        # Reference metrics can still be checked, but per-sample prediction matching cannot.
        status, reason = "ambiguous_provenance", "test_predictions.npy absent; reference_unavailable"
    train_seconds, train_detail, train_status = training_time_info(source)
    evidence = {
        "model": model, "dataset": dataset, "seed": seed_from_path(task_root), "attempt": str(source.resolve()),
        "config_sha256": sha256_file(config_path) if config_path else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "data_config_sha256": sha256_file(data_path) if data_path else None,
    }
    return {
        "model": model, "display_model": MODELS[model], "dataset": dataset, "display_dataset": DISPLAY_DATASET[dataset],
        "seed": seed_from_path(task_root), "task_root": str(task_root.resolve()), "attempt_path": str(source.resolve()),
        "association": association, "identity_evidence": (
            "saved model_name" if effective_name is not None else "exact formal result root plus saved model configuration without model_name"
        ), "saved_model_name": effective_name or "",
        "config_path": str(config_path.resolve()) if config_path else "",
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint else "", "preprocess_config_path": str(data_path.resolve()) if data_path else "",
        "preprocess_cache_candidates": ";".join(str(path.resolve()) for path in caches),
        "prediction_path": str((source / "prediction.json").resolve()) if (source / "prediction.json").is_file() else "",
        "saved_predictions_path": str(saved_predictions.resolve()) if saved_predictions.is_file() else "",
        "test_rmse": (prediction or {}).get("metrics", {}).get("rmse"), "status": status, "reason": reason,
        "training_seconds": train_seconds, "training_time_detail": train_detail, "training_time_status": train_status,
        "artifact_fingerprint": fingerprint(evidence),
    }


def hinge_records(results: Path, dataset: str) -> list[dict[str, Any]]:
    root = results / "hingemix_ablation_v2"
    manifest = read_json(root / "manifest.json") or {}
    records = []
    for task in manifest.get("tasks", []):
        if task.get("dataset") != dataset or task.get("ablation") != "full" or task.get("model") != "hingemix_ablation_v2":
            continue
        task_root = root / "full" / dataset / f"seed_{task.get('seed')}"
        attempt = selected_attempt(task_root)
        record = make_record("hingemix", dataset, task_root, attempt, "formal_hingemix_ablation_v2/full manifest + selected marker")
        if task.get("data_seed") != 42:
            record.update(status="configuration_mismatch", reason="formal task data_seed is not 42")
        records.append(record)
    if not records:
        missing_root = root / "full" / dataset
        records.append(make_record("hingemix", dataset, missing_root, None, "formal hingemix_ablation_v2/full manifest task not found"))
    return records


def baseline_records(results: Path, model: str, dataset: str) -> list[dict[str, Any]]:
    root = results / model / dataset
    compare_sources = compare92_sources(results, model, dataset)
    if not root.exists():
        if compare_sources:
            return [
                make_record(model, dataset, source, selected_attempt(source) or source, "compare92 exact registry/dataset row source path")
                for source in compare_sources
            ]
        return [make_record(model, dataset, root, None, "expected formal root absent")]
    roots: list[Path] = []
    if (root / "prediction.json").is_file() or (root / "parameter_result.json").is_file():
        roots.append(root)
    roots.extend(sorted(path for path in root.glob("seed_*") if path.is_dir()))
    if not roots:
        return [make_record(model, dataset, root, None, "no direct or seed result root")]
    records = []
    for task_root in roots:
        attempt = selected_attempt(task_root) or (task_root if (task_root / "prediction.json").is_file() else None)
        records.append(make_record(model, dataset, task_root, attempt, "expected formal root; config/checkpoint evidence required"))
    return records


def compare92_sources(results: Path, model: str, dataset: str) -> list[Path]:
    """Use only explicit, exact registry/dataset references from compare92 CSVs."""
    root = results / "compare92"
    found: dict[Path, None] = {}
    if not root.is_dir():
        return []
    for csv_path in root.glob("*.csv"):
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = csv.DictReader(handle)
                for row in rows:
                    source_model = row.get("model") or row.get("model_name")
                    source_dataset = row.get("dataset") or row.get("dataset_name")
                    if source_model != model or source_dataset != dataset:
                        continue
                    value = next((row.get(name) for name in ("attempt_path", "source_path", "output_dir", "result_path") if row.get(name)), None)
                    if not value:
                        continue
                    path = Path(value)
                    if path.is_file():
                        path = path.parent
                    if path.is_dir():
                        found[path.resolve()] = None
        except (OSError, csv.Error, UnicodeError):
            continue
    return sorted(found)


def environment() -> dict[str, Any]:
    result = {"timestamp": dt.datetime.now().isoformat(), "platform": platform.platform(), "processor": platform.processor(), "python": sys.version, "cwd": str(Path.cwd().resolve())}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, timeout=10)
        result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        result["git_revision"] = "unavailable"
    try:
        import numpy
        result["numpy"] = numpy.__version__
    except ImportError:
        result["numpy"] = "unavailable"
    try:
        import torch
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
    except ImportError:
        result["torch"] = "unavailable"
    for command, key in ((["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], "gpu"), (["lscpu"], "lscpu")):
        try:
            capture = subprocess.run(command, text=True, capture_output=True, timeout=10)
            result[key] = capture.stdout.strip() if capture.returncode == 0 else f"unavailable: {capture.stderr.strip()}"
        except (OSError, subprocess.SubprocessError):
            result[key] = "unavailable"
    return result


def main() -> None:
    args = parse_args()
    results = Path(args.results_root).resolve()
    stamp = dt.datetime.now().strftime("audit_%Y%m%d_%H%M%S")
    output = Path(args.output).resolve() if args.output else Path(args.output_root).resolve() / stamp
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            rows.extend(hinge_records(results, dataset) if model == "hingemix" else baseline_records(results, model, dataset))
    fields = list(rows[0]) if rows else ["model", "dataset", "seed", "status", "reason"]
    write_csv(output / "inventory.csv", rows, fields)
    training = [row for row in rows if row["training_time_status"] != "complete"]
    write_csv(output / "training_time_remeasure.csv", training, fields)
    write_json(output / "environment.json", environment())
    status_counts = {name: sum(row["status"] == name for row in rows) for name in sorted({row["status"] for row in rows})}
    lines = ["# HingeMix Efficiency Artifact Audit", "", f"- Results root: `{results}`", f"- Results root exists: `{results.is_dir()}`", f"- Records: {len(rows)}", f"- Statuses: `{status_counts}`", "", "## Protocol", "", "- `ready` requires a selected artifact, matching registered model name when that field was saved, saved configuration, checkpoint, saved transformed Dataset cache, final prediction record, and per-sample predictions.", "- A baseline configuration without `model_name` is associated only through its exact formal result root; this evidence is recorded in `identity_evidence`.", "- Historical cached datasets enable `model_only` recovery. They do not retain fitted preprocessing transformer objects for strict raw-input end-to-end timing.", "- Training times are audit evidence only; none is inferred from timestamps or epoch counts."]
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} inventory records to {output}")
    print("statuses:", status_counts)


if __name__ == "__main__":
    main()
