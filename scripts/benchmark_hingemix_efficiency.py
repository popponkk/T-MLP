#!/usr/bin/env python3
"""Measure restored formal artifacts without training or modifying source results.

Historical artifacts in this project retain transformed Dataset caches rather
than fitted transform objects.  Therefore this tool measures only ``model_only``
(preprocessed features -> model -> target inverse transform) unless a future
artifact adapter provides a fitted preprocessor.  It never relabels that scope
as end-to-end.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Running ``python scripts/...`` otherwise exposes only ``scripts/`` and not
# project packages such as ``utils`` and ``models``.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_efficiency_common import (  # noqa: E402
    MODELS, cache_candidates, fingerprint, read_yaml, sha256_file, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="inventory.csv from audit_hingemix_efficiency.py")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--scope", choices=("model_only", "end_to_end"), default="model_only")
    parser.add_argument("--output-root", default="results/hingemix_efficiency")
    parser.add_argument("--resume", action="store_true", help="Skip only a completed task with the identical artifact and protocol identity")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    # ``utf-8-sig`` accepts both POSIX UTF-8 output and Excel/PowerShell CSVs
    # that carry a UTF-8 BOM.
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pick_record(args: argparse.Namespace) -> dict[str, str]:
    candidates = [row for row in rows(Path(args.inventory)) if row["model"] == args.model and row["dataset"] == args.dataset and row["status"] == "ready"]
    if args.seed is not None:
        candidates = [row for row in candidates if row["seed"] == str(args.seed)]
    if len(candidates) != 1:
        raise SystemExit(f"Expected exactly one ready artifact, found {len(candidates)}. Use --seed or inspect inventory.csv.")
    return candidates[0]


def load_cached_dataset(record: dict[str, str]):
    from utils.data_utils import load_pickle
    data_config = Path(record["preprocess_config_path"])
    document = read_yaml(data_config) or {}
    expected = document.get("transformation")
    candidates = cache_candidates(data_config)
    matches = []
    for path in candidates:
        try:
            transformation, dataset = load_pickle(path)
        except Exception:
            continue
        if isinstance(expected, dict) and vars(transformation) == expected:
            matches.append((path, dataset))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cache matching data_config transformation; found {len(matches)}")
    return matches[0]


def restore_model(record: dict[str, str], dataset, device: str):
    import torch
    from utils.model_utils import make_baseline
    config_doc = read_yaml(Path(record["config_path"])) or {}
    saved_model = config_doc.get("model") if isinstance(config_doc.get("model"), dict) else config_doc
    if record["model"] == "hingemix":
        # ``configs.yaml`` is post-construction metadata for the v2 ablation
        # class.  Its outer keys describe the selected ablation and must not be
        # forwarded as constructor kwargs; the original constructor input is
        # retained under requested_model_config.
        requested = saved_model.get("requested_model_config") if isinstance(saved_model, dict) else None
        if not isinstance(requested, dict):
            raise RuntimeError("HingeMix saved config has no requested_model_config for faithful restoration")
        model_config = dict(requested)
        model_config["ablation"] = saved_model.get("ablation", "full")
    else:
        model_config = saved_model
    if not isinstance(model_config, dict):
        raise RuntimeError("Saved model configuration has no model mapping")
    category_sizes = dataset.get_category_sizes("train")
    categories = category_sizes or None
    n_labels = dataset.n_classes or 1
    registry = "hingemix_ablation_v2" if record["model"] == "hingemix" else record["model"]
    model = make_baseline(registry, dict(model_config), dataset.n_num_features, categories, n_labels, device=device, dataset=dataset)
    model.load_best_dnn(record["attempt_path"])
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model.model.eval()
    return model


def input_arrays(dataset):
    x_num = dataset.X_num["test"] if dataset.X_num is not None else None
    x_cat = dataset.X_cat["test"] if dataset.X_cat is not None else None
    if x_num is None and x_cat is None:
        raise RuntimeError("Test split has no usable model features")
    return x_num, x_cat


def prediction_fn(model, x_num, x_cat, device: str, y_info: dict[str, Any]):
    import torch
    is_dnn = isinstance(getattr(model, "model", None), torch.nn.Module)
    y_mean, y_std = float(y_info.get("mean", 0.0)), float(y_info.get("std", 1.0))
    if is_dnn:
        target_device = torch.device(device)
        def run(num, cat):
            num_tensor = None if num is None else torch.as_tensor(num, dtype=torch.float32, device=target_device)
            cat_tensor = None if cat is None else torch.as_tensor(cat, device=target_device)
            with torch.no_grad():
                value = model.model(num_tensor, cat_tensor)
            return value.detach().float().cpu().numpy().reshape(-1) * y_std + y_mean
    else:
        def run(num, cat):
            features = cat if num is None else num if cat is None else np.concatenate([num, cat], axis=1)
            return np.asarray(model.model.predict(features)).reshape(-1) * y_std + y_mean
    return run, is_dnn


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def process_snapshot(device: str) -> str:
    if device != "cuda":
        return "not_applicable"
    try:
        run = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True, capture_output=True, timeout=10)
        return run.stdout.strip() if run.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def runtime_environment(torch, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(), "processor": platform.processor(),
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if device == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, timeout=10)
        result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        result["git_revision"] = "unavailable"
    return result


def sizes(record: dict[str, str], selected_cache: Path) -> dict[str, Any]:
    paths = [Path(record[name]) for name in ("checkpoint_path", "config_path", "preprocess_config_path") if record.get(name)]
    # Only the cache used for this restoration is an inference artifact.  Other
    # caches in the same directory belong to other preprocessing variants.
    paths.append(selected_cache)
    unique = {path.resolve() for path in paths if path.is_file()}
    model_file = Path(record["checkpoint_path"])
    return {"model_file_bytes": model_file.stat().st_size, "model_file_mib": model_file.stat().st_size / 2**20, "inference_artifact_bytes": sum(path.stat().st_size for path in unique), "inference_artifact_mib": sum(path.stat().st_size for path in unique) / 2**20}


def main() -> None:
    args = parse_args()
    if args.scope == "end_to_end":
        raise SystemExit("end_to_end is unavailable for historical artifacts: fitted preprocessing objects were not saved. Use --scope model_only.")
    if args.threads < 1 or args.warmup < 0 or args.rounds < 1 or args.requests < 1 or any(value < 1 for value in args.batch_sizes):
        raise SystemExit("threads, batch sizes, rounds, and requests must be positive")
    record = pick_record(args)
    if args.dry_run:
        print("selected:", record["model"], record["dataset"], f"seed={record['seed']}", record["attempt_path"])
        print("scope=model_only; transformed Dataset cache is required; no training or result-file writes occur")
        return
    os.environ["OMP_NUM_THREADS"] = str(args.threads); os.environ["MKL_NUM_THREADS"] = str(args.threads); os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but unavailable")
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    cache_path, dataset = load_cached_dataset(record)
    identity = {
        "version": 2, "artifact_fingerprint": record["artifact_fingerprint"],
        "cache_sha256": sha256_file(cache_path), "model": record["model"],
        "dataset": record["dataset"], "seed": record["seed"], "device": args.device,
        "gpu": args.gpu if args.device == "cuda" else None, "scope": "model_only",
        "threads": args.threads, "batch_sizes": args.batch_sizes, "warmup": args.warmup,
        "rounds": args.rounds, "requests": args.requests,
    }
    identity_fingerprint = fingerprint(identity)
    output = Path(args.output_root).resolve() / "tasks" / (
        f"{record['model']}__{record['dataset']}__seed_{record['seed']}__{args.device}__{identity_fingerprint[:16]}"
    )
    if output.exists():
        # protocol.json is JSON; compare the immutable identity, never a timestamp.
        try:
            existing = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if args.resume and isinstance(existing, dict) and existing.get("measurement_identity") == identity:
            print(f"already complete with matching identity: {output}")
            return
        raise SystemExit(f"measurement output already exists and is not resumable with this identity: {output}")
    model = restore_model(record, dataset, device)
    if not isinstance(getattr(model, "model", None), torch.nn.Module) and hasattr(model.model, "set_params"):
        try:
            model.model.set_params(n_jobs=args.threads)
        except (TypeError, ValueError):
            pass
    x_num, x_cat = input_arrays(dataset)
    test_size = len(x_num) if x_num is not None else len(x_cat)
    saved_path = Path(record["saved_predictions_path"])
    run, is_dnn = prediction_fn(model, x_num, x_cat, device, dataset.y_info)
    # Consistency uses the same full test order but is not part of timing.
    restored = run(x_num, x_cat)
    saved = np.load(saved_path).reshape(-1)
    # Saved project predictions are standardized; transform them only for comparison.
    saved_raw = saved * float(dataset.y_info.get("std", 1.0)) + float(dataset.y_info.get("mean", 0.0))
    max_abs = float(np.max(np.abs(restored - saved_raw))) if len(saved_raw) else 0.0
    if len(restored) != len(saved_raw) or not np.allclose(restored, saved_raw, rtol=1e-5, atol=1e-6):
        raise SystemExit(f"prediction_mismatch: restored output differs from saved prediction; max_abs_diff={max_abs}")
    pre_gpu, raw_rows, summaries = process_snapshot(args.device), [], []
    contamination = "potentially_contaminated" if args.device == "cuda" and pre_gpu not in {"", "not_applicable", "unavailable"} else "not_detected"
    for batch_size in args.batch_sizes:
        if test_size < batch_size:
            summaries.append({"model": record["model"], "dataset": record["dataset"], "seed": record["seed"], "device": args.device, "scope": "model_only", "batch_size": batch_size, "status": "unavailable_insufficient_test_samples", "test_size": test_size, "contamination_status": contamination})
            continue
        # Fixed, ordered requests: positions are prepared outside timing and repeat deterministically.
        batches = [np.arange(offset, offset + batch_size) for offset in range(0, test_size - batch_size + 1, batch_size)]
        for index in range(args.warmup):
            positions = batches[index % len(batches)]
            run(None if x_num is None else x_num[positions], None if x_cat is None else x_cat[positions])
        synchronize(args.device)
        durations = []
        for timing_round in range(args.rounds):
            for request in range(args.requests):
                positions = batches[request % len(batches)]
                before = time.perf_counter_ns()
                _ = run(None if x_num is None else x_num[positions], None if x_cat is None else x_cat[positions])
                synchronize(args.device)
                duration = (time.perf_counter_ns() - before) / 1e6
                durations.append(duration)
                raw_rows.append({"model": record["model"], "dataset": record["dataset"], "seed": record["seed"], "device": args.device, "scope": "model_only", "batch_size": batch_size, "timing_round": timing_round, "request": request, "latency_ms": duration, "request_positions": ",".join(map(str, positions.tolist()))})
        summaries.append({"model": record["model"], "dataset": record["dataset"], "seed": record["seed"], "device": args.device, "scope": "model_only", "batch_size": batch_size, "status": "measured", "n_requests": len(durations), "n_samples": len(durations) * batch_size, "latency_p50_ms": float(np.percentile(durations, 50)), "latency_p95_ms": float(np.percentile(durations, 95)), "mean_request_latency_ms": float(np.mean(durations)), "throughput_samples_per_second": float(len(durations) * batch_size / (sum(durations) / 1000.0)), "batch_average_sample_cost_ms": float(np.mean(durations) / batch_size), "test_size": test_size, "request_sequence_reused": True, "contamination_status": contamination})
    post_gpu = process_snapshot(args.device)
    model_bytes = sizes(record, cache_path)
    parameters = {"parameter_count": "N/A", "trainable_parameter_count": "N/A"}
    if is_dnn:
        parameters = {"parameter_count": sum(item.numel() for item in model.model.parameters()), "trainable_parameter_count": sum(item.numel() for item in model.model.parameters() if item.requires_grad)}
    protocol = {"version": 2, "measurement_identity": identity, "scope": "model_only", "device": device, "threads_requested": args.threads, "torch_threads_effective": torch.get_num_threads(), "batch_sizes": args.batch_sizes, "warmup": args.warmup, "rounds": args.rounds, "requests_per_round": args.requests, "input": "saved transformed test features -> dtype/device conversion -> model -> target inverse transform -> CPU readable prediction", "excluded": "disk IO, checkpoint loading, model construction, metric calculation, output writes", "pre_gpu_processes": pre_gpu, "post_gpu_processes": post_gpu, "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if args.device == "cuda" else None, "artifact_fingerprint": record["artifact_fingerprint"], "cache_path": str(cache_path.resolve()), "environment": runtime_environment(torch, args.device)}
    write_csv(output / "raw_request_times.csv", raw_rows, ["model", "dataset", "seed", "device", "scope", "batch_size", "timing_round", "request", "latency_ms", "request_positions"])
    summaries = [{**row, **model_bytes, **parameters, "prediction_max_abs_diff": max_abs, "protocol_fingerprint": fingerprint(protocol)} for row in summaries]
    fields = sorted({key for row in summaries for key in row})
    write_csv(output / "summary.csv", summaries, fields)
    write_json(output / "protocol.json", protocol)
    write_json(output / "verification.json", {"reference": str(saved_path.resolve()), "max_abs_difference": max_abs, "status": "verified"})
    print(f"Wrote {len(raw_rows)} request timings to {output}")


if __name__ == "__main__":
    main()
