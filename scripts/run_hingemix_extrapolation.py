#!/usr/bin/env python3
"""Run resumable HingeMix configuration-extrapolation tasks from a prepared manifest."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_extrapolation_common import FORMAL_HINGEMIX, fingerprint, load_yaml, read_json, write_csv, write_json  # noqa: E402


SUPPORTED_MODELS = ("hingemix", "lightgbm", "mlp")
MODEL_LABELS = {"hingemix": "HingeMix", "lightgbm": "LightGBM", "mlp": "MLP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Same plan passed to prepare_hingemix_extrapolation.py")
    parser.add_argument("--manifest", default=None, help="Prepared manifest path; defaults from plan/output root")
    parser.add_argument("--models", nargs="+", default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", default="results/hingemix_extrapolation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=2048)
    return parser.parse_args()


def atomic_task(path: Path, value: dict[str, Any]) -> None:
    write_json(path, value)


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def model_profile(repo: Path, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = repo / "configs" / "default" / f"{model}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No default config for requested extrapolation model: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("model", {})
    config.setdefault("training", config.get("fit", {}))
    cli: dict[str, Any] = {}
    if model == "hingemix":
        config["model"].update(FORMAL_HINGEMIX)
        config["model"]["model_name"] = "hingemix"
        config["training"].update(lr=1e-5, batch_size=32)
        cli = {"lr": 1e-5, "batch_size": 32, "d_token": 1024, "n_layers": 1}
    elif model == "mlp":
        # main.py otherwise overwrites non-formal DNN LR with its CLI default.
        cli = {"lr": float(config["training"].get("lr", 1e-3)), "batch_size": 32}
    return config, cli


def task_dir(root: Path, manifest: dict[str, Any], model: str, seed: int) -> Path:
    plan = manifest["plan"]
    return root / manifest["experiment_id"] / plan["source"]["dataset"] / plan["extrapolation"]["parameter"] / model / f"seed_{seed}"


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    values = [int(path.name.rsplit("_", 1)[1]) for path in attempts.glob("attempt_*") if path.name.rsplit("_", 1)[-1].isdigit()]
    return attempts / f"attempt_{max(values, default=0) + 1:03d}"


def lock_active(path: Path) -> bool:
    try:
        os.kill(int(path.read_text(encoding="utf-8").strip()), 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError):
        return True


def acquire(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".task.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if lock_active(lock):
            return None
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    return lock, fd


def release(item) -> None:
    if item:
        path, fd = item
        os.close(fd)
        path.unlink(missing_ok=True)


def metrics_valid(prediction: dict[str, Any] | None) -> bool:
    metrics = (prediction or {}).get("metrics") or {}
    return all(isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]) for key in ("rmse", "mae"))


def valid_result(root: Path, task: dict[str, Any]) -> tuple[bool, str]:
    marker = read_json(root / "parameter_result.json")
    if not isinstance(marker, dict) or marker.get("task_fingerprint") != task["task_fingerprint"]:
        return False, "missing or incompatible task marker"
    attempt = Path(str(marker.get("attempt_dir", "")))
    task_json = read_json(attempt / "task.json")
    prediction = read_json(attempt / "prediction.json")
    if not isinstance(task_json, dict) or task_json.get("task_fingerprint") != task["task_fingerprint"]:
        return False, "attempt metadata mismatch"
    if not metrics_valid(prediction):
        return False, "final test RMSE/MAE missing"
    if not (attempt / "per_test_configuration.csv").is_file():
        return False, "per-test-configuration metrics missing"
    return True, ""


def materialize(repo: Path, root: Path, task: dict[str, Any]) -> tuple[Path, Path, Path]:
    attempt = next_attempt(root)
    attempt.mkdir(parents=True)
    config, _ = model_profile(repo, task["model"])
    if task["model"] == "hingemix":
        config["model"]["breakpoint_cache_dir"] = str((attempt / "breakpoint_cache").resolve())
    config_path = attempt / "final_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    preprocess = attempt / "preprocessing_cache"
    preprocess.mkdir()
    atomic_task(attempt / "task.json", {**task, "attempt_dir": str(attempt.resolve()), "preprocess_cache_dir": str(preprocess.resolve())})
    return attempt, config_path, preprocess


def build_command(repo: Path, task: dict[str, Any], attempt: Path, config: Path, preprocess: Path, gpu: int) -> list[str]:
    _, cli = model_profile(repo, task["model"])
    command = [
        sys.executable, "-u", "main.py", "--model", task["model"], "--dataset", task["dataset"],
        "--device", "cuda", "--gpu", str(gpu), "--seed", str(task["seed"]),
        "--data_seed", str(task["data_seed"]), "--config", str(config), "--output_dir", str(attempt),
        "--preprocess-cache-dir", str(preprocess), "--no-test-during-fit", "--save-evaluation-arrays",
        "--strict-train-preprocessing",
    ]
    for key, value in cli.items():
        command.extend([f"--{key}", str(value)])
    return command


def postprocess(attempt: Path, manifest_dir: Path) -> None:
    arrays = np.load(attempt / "test_evaluation_arrays.npz")
    prediction, target = arrays["prediction"].reshape(-1), arrays["target"].reshape(-1)
    with (manifest_dir / "samples.csv").open("r", newline="", encoding="utf-8") as handle:
        test_rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(test_rows) != len(prediction) or len(target) != len(prediction):
        raise ValueError("Saved test arrays and manifest test rows have different lengths")
    rows = []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for metadata, pred, truth in zip(test_rows, prediction, target):
        if not (np.isfinite(pred) and np.isfinite(truth)):
            raise ValueError("Non-finite final test prediction/target")
        error = float(pred - truth)
        rows.append({"sample_id": metadata["sample_id"], "configuration_value": metadata["configuration_value"], "y_true": float(truth), "y_pred": float(pred), "abs_error": abs(error), "squared_error": error * error})
        grouped.setdefault(metadata["configuration_value"], []).append((float(pred), float(truth)))
    write_csv(attempt / "test_predictions.csv", rows, list(rows[0]))
    metrics = []
    for value, pairs in sorted(grouped.items(), key=lambda item: float(item[0])):
        error = np.asarray([pred - truth for pred, truth in pairs])
        metrics.append({"configuration_value": value, "n_samples": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))})
    write_csv(attempt / "per_test_configuration.csv", metrics, ["configuration_value", "n_samples", "rmse", "mae"])


def build_tasks(repo: Path, manifest: dict[str, Any], models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    plan = manifest["plan"]
    tasks = []
    for model in models:
        config, cli = model_profile(repo, model)
        for seed in seeds:
            identity = {"model": model, "dataset": manifest["derived_dataset_name"], "seed": seed, "data_seed": plan["extrapolation"].get("split_seed", 42), "manifest_fingerprint": manifest["manifest_fingerprint"], "config": config, "cli": cli}
            tasks.append({**identity, "task_fingerprint": fingerprint(identity), "status": "pending"})
    return tasks


def check_gpus(gpus: list[int], min_free: int) -> None:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"nvidia-smi failed: {result.stderr.strip()}")
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines()}
    bad = [gpu for gpu in gpus if free.get(gpu, 0) < min_free]
    print(f"GPU free memory (MiB): {free}")
    if bad:
        raise SystemExit(f"Refusing to start; insufficient free GPU memory on {bad}")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load_yaml(Path(args.plan).resolve())
    experiment = safe_name(str(plan.get("experiment_id") or f"{plan['source']['dataset']}_{plan['extrapolation']['parameter']}_extrapolation"))
    output_root = Path(args.output_root).resolve()
    manifest_dir = Path(args.manifest).resolve().parent if args.manifest else output_root / "manifests" / experiment
    manifest_path = Path(args.manifest).resolve() if args.manifest else manifest_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Prepared manifest not found: {manifest_path}; run prepare_hingemix_extrapolation.py first")
    view_info = read_json(Path(manifest["derived_dataset_dir"]) / "info.json")
    if not isinstance(view_info, dict) or view_info.get("extrapolation_manifest_fingerprint") != manifest.get("manifest_fingerprint"):
        raise SystemExit("Derived dataset info does not match the prepared manifest")
    models = args.models or list(plan.get("models") or ["hingemix", "lightgbm", "mlp"])
    seeds = args.seeds or list(plan.get("training_seeds") or [0, 1, 2, 3, 4])
    if any(model not in SUPPORTED_MODELS for model in models) or len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise SystemExit("Unsupported/duplicate model or duplicate seed")
    tasks = build_tasks(repo, manifest, models, seeds)
    task_manifest = output_root / experiment / "tasks.json"
    prior = read_json(task_manifest)
    if isinstance(prior, dict) and prior.get("manifest_fingerprint") not in (None, manifest["manifest_fingerprint"]):
        raise SystemExit("Existing task manifest belongs to another prepared split")
    statuses: dict[str, int] = {}
    pending = []
    for task in tasks:
        root = task_dir(output_root, manifest, task["model"], task["seed"])
        valid, reason = valid_result(root, task)
        if valid:
            task["status"] = "completed"
        elif (root / ".task.lock").exists() and lock_active(root / ".task.lock"):
            task["status"] = "running"
        elif (root / "parameter_result.json").exists():
            task.update(status="needs_review", reason=reason)
        elif any((root / "attempts").glob("attempt_*")) and not args.retry_failed:
            task.update(status="failed", reason="prior attempt exists; pass --retry-failed to create a new attempt")
        else:
            pending.append(task)
        statuses[task["status"]] = statuses.get(task["status"], 0) + 1
    write_json(task_manifest, {"schema_version": 1, "manifest_fingerprint": manifest["manifest_fingerprint"], "tasks": tasks, "statuses": statuses})
    print(f"design: {len(models)} models x 1 prepared extrapolation split x {len(seeds)} seeds = {len(tasks)}")
    print(f"experiment={experiment} dataset={manifest['derived_dataset_name']} parameter={plan['extrapolation']['parameter']}")
    print(f"states: {statuses}; pending: {len(pending)}")
    for task in pending:
        print(f"pending: {task['model']} seed={task['seed']} {task_dir(output_root, manifest, task['model'], task['seed'])}")
    if args.dry_run:
        return
    check_gpus(args.gpus, args.min_free_mib)
    children: dict[int, Any] = {}
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        for process, *_rest in children.values():
            process.terminate()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while (pending and not stopping) or children:
        busy = {item[3] for item in children.values()}
        free = [gpu for gpu in args.gpus if gpu not in busy]
        while free and pending and not stopping:
            task, gpu = pending.pop(0), free.pop(0)
            root = task_dir(output_root, manifest, task["model"], task["seed"])
            held = acquire(root)
            if held is None:
                task.update(status="needs_review", reason="active task lock")
                continue
            attempt, config, preprocess = materialize(repo, root, task)
            task.update(status="running", attempt_dir=str(attempt.resolve()), gpu=gpu, started_at=time.time())
            write_json(task_manifest, {"schema_version": 1, "manifest_fingerprint": manifest["manifest_fingerprint"], "tasks": tasks})
            log = (attempt / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(build_command(repo, task, attempt, config, preprocess, gpu), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, task, held, gpu, log)
        for pid, (process, task, held, gpu, log) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); release(held); children.pop(pid)
            task.update(finished_at=time.time(), return_code=code)
            try:
                if code == 0:
                    postprocess(Path(task["attempt_dir"]), manifest_dir)
                    prediction = read_json(Path(task["attempt_dir"]) / "prediction.json")
                    if not metrics_valid(prediction):
                        raise ValueError("invalid final test metrics")
                    marker = {**task, "attempt_dir": task["attempt_dir"], "test_rmse": prediction["metrics"]["rmse"], "test_mae": prediction["metrics"]["mae"]}
                    atomic_task(task_dir(output_root, manifest, task["model"], task["seed"]) / "parameter_result.json", marker)
                    task.update(status="completed", reason=None)
                else:
                    task.update(status="failed", reason=f"main.py exited with {code}")
            except Exception as error:
                task.update(status="failed", reason=str(error))
            print(f"{task['status']}: gpu={gpu} {task['model']} seed={task['seed']}", flush=True)
            write_json(task_manifest, {"schema_version": 1, "manifest_fingerprint": manifest["manifest_fingerprint"], "tasks": tasks})
        if children:
            time.sleep(0.5)
    if stopping:
        raise SystemExit("Interrupted; only child processes created by this runner were terminated")


if __name__ == "__main__":
    main()
