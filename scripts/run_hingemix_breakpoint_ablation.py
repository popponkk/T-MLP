#!/usr/bin/env python3
"""Run isolated HingeMix breakpoint-initialization comparison tasks.

The runner never changes saved data splits.  It creates separate preprocessing
and breakpoint caches for every strategy/dataset/seed attempt and uses the
existing main.py training, validation checkpoint selection, and final test flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
STRATEGIES = ("gbdt", "quantile", "uniform")
SEEDS = (0, 1, 2, 3, 4)
FORMAL_MODEL = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
    "learnable_breakpoints": True,
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_dataset(repo: Path, dataset: str) -> Path:
    candidates = [repo / "data" / "datasets" / dataset, repo / "data" / "custom_datasets" / dataset]
    found = [path for path in candidates if (path / "info.json").is_file()]
    if not found:
        raise FileNotFoundError(f"No registered dataset directory for {dataset}: {candidates}")
    if len(found) > 1:
        print(f"warning: {dataset} exists in multiple data roots; using {found[0]}", flush=True)
    return found[0]


def feature_names(directory: Path, n_features: int) -> tuple[list[str], str]:
    info = read_json(directory / "info.json") or {}
    for key in ("num_feature_names", "numerical_feature_names", "numeric_feature_names", "feature_names"):
        values = info.get(key)
        if isinstance(values, list) and len(values) == n_features and all(isinstance(x, str) for x in values):
            return values, f"info.json:{key}"
    return [f"feature_{index}" for index in range(n_features)], "generated_numeric_feature_indices"


def dataset_evidence(repo: Path, dataset: str) -> dict[str, Any]:
    directory = resolve_dataset(repo, dataset)
    required = [directory / "info.json", directory / "X_num_train.npy", directory / "y_train.npy"]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{dataset} lacks required saved split files: {absent}")
    x_train = np.load(directory / "X_num_train.npy", mmap_mode="r")
    y_train = np.load(directory / "y_train.npy", mmap_mode="r")
    if len(x_train) != len(y_train):
        raise ValueError(f"{dataset} train feature/target lengths differ")
    names, name_source = feature_names(directory, int(x_train.shape[1]))
    files = sorted(directory.glob("*.npy")) + [directory / "info.json"]
    return {
        "dataset_dir": str(directory.resolve()),
        "train_size": int(len(y_train)),
        "n_numeric_features": int(x_train.shape[1]),
        "feature_names": names,
        "feature_name_source": name_source,
        "dataset_files_sha256": {path.name: sha256_file(path) for path in files if path.is_file()},
    }


def source_fingerprint(repo: Path) -> dict[str, str]:
    paths = (
        repo / "models" / "hingemix.py", repo / "models" / "ggpl_tmlp.py",
        repo / "data" / "processor.py", repo / "models" / "abstract.py", repo / "main.py",
        repo / "configs" / "default" / "hingemix_breakpoint_ablation.yaml",
        repo / "scripts" / "run_hingemix_breakpoint_ablation.py",
    )
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch smoke tasks under a separate output root.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", default="results/hingemix_breakpoint_ablation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    return parser.parse_args()


def task_root(root: Path, task: dict[str, Any]) -> Path:
    return root / task["strategy"] / task["dataset"] / f"seed_{task['training_seed']}"


def build_manifest(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True).stdout.strip() or None
    code = source_fingerprint(repo)
    smoke = bool(args.smoke)
    selected_datasets = ("hpcg2",) if smoke else tuple(args.datasets)
    selected_strategies = tuple(args.strategies)
    selected_seeds = (0,) if smoke else tuple(args.seeds)
    tasks: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        evidence = dataset_evidence(repo, dataset)
        for strategy in selected_strategies:
            for training_seed in selected_seeds:
                identity = {
                    "model": "hingemix", "dataset": dataset, "strategy": strategy,
                    "training_seed": training_seed, "data_seed": args.data_seed,
                    "formal_model_parameters": FORMAL_MODEL,
                    "training": {"lr": args.lr, "batch_size": args.batch_size, "max_epochs": 2 if smoke else None},
                    "dataset_evidence": evidence, "source_fingerprint": code,
                    "git_revision": revision, "smoke": smoke,
                    "test_during_training": False,
                }
                tasks.append({**identity, "config_fingerprint": fingerprint(identity), "status": "pending"})
    return {
        "schema_version": 1, "experiment": "hingemix_breakpoint_ablation",
        "smoke": smoke, "created_at": time.time(), "expected_tasks": len(tasks),
        "datasets": list(selected_datasets), "strategies": list(selected_strategies),
        "training_seeds": list(selected_seeds), "data_seed": args.data_seed,
        "formal_model_parameters": FORMAL_MODEL, "tasks": tasks,
    }


def valid_result(root: Path, task: dict[str, Any]) -> tuple[bool, str | None]:
    marker = read_json(root / "parameter_result.json")
    if not marker or marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "missing or conflicting selected result"
    attempt = Path(marker.get("attempt_dir", ""))
    metadata, prediction = read_json(attempt / "task.json"), read_json(attempt / "prediction.json")
    metrics = (prediction or {}).get("metrics", {})
    if not metadata or metadata.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt metadata fingerprint mismatch"
    if not all(isinstance(metrics.get(key), (int, float)) and np.isfinite(metrics[key]) for key in ("rmse", "mae", "r2")):
        return False, "final test RMSE/MAE/R2 is missing or invalid"
    if not (attempt / "breakpoint_diagnostics.json").is_file():
        return False, "breakpoint diagnostics missing"
    if not (attempt / "test_prediction_trace.npz").is_file():
        return False, "paired final-test prediction trace missing"
    return True, None


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.rsplit("_", 1)[1]) for path in attempts.glob("attempt_*") if path.name.rsplit("_", 1)[-1].isdigit()]
    return attempts / f"attempt_{max(numbers, default=0) + 1:03d}"


def lock_is_active(path: Path) -> bool:
    try:
        os.kill(int(path.read_text(encoding="utf-8").strip()), 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError):
        return True
    return True


def acquire_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".task.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if lock_is_active(path):
            return None
        path.unlink(missing_ok=True)
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode("utf-8"))
    return path, handle


def release_lock(item) -> None:
    if item is not None:
        path, handle = item
        os.close(handle)
        path.unlink(missing_ok=True)


def materialize(repo: Path, root: Path, task: dict[str, Any]) -> tuple[Path, Path, Path]:
    attempt = next_attempt(root)
    attempt.mkdir(parents=True)
    preprocess_cache = attempt / "preprocessing_cache"
    preprocess_cache.mkdir()
    breakpoint_cache = attempt / "breakpoint_cache"
    config = yaml.safe_load((repo / "configs" / "default" / "hingemix_breakpoint_ablation.yaml").read_text(encoding="utf-8"))
    config["model"].update(FORMAL_MODEL)
    config["model"].update({
        "model_name": "hingemix", "breakpoint_strategy": task["strategy"],
        "breakpoint_init": "gbdt", "breakpoint_fallback": "quantile",
        "breakpoint_cache_dir": str(breakpoint_cache),
        "breakpoint_cache_key": task["config_fingerprint"],
    })
    config["training"].update(lr=task["training"]["lr"], batch_size=task["training"]["batch_size"])
    if task["smoke"]:
        config["training"]["max_epochs"] = task["training"]["max_epochs"]
        config["training"]["patience"] = 0
    final_config = attempt / "final_config.yaml"
    final_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(preprocess_cache / "experiment.json", {
        "experiment": "hingemix_breakpoint_ablation", "config_fingerprint": task["config_fingerprint"],
        "dataset_evidence": task["dataset_evidence"], "strategy": task["strategy"],
    })
    atomic_json(attempt / "task.json", {
        **task, "attempt_dir": str(attempt.resolve()),
        "preprocess_cache_dir": str(preprocess_cache.resolve()),
        "breakpoint_cache_dir": str(breakpoint_cache.resolve()),
        "effective_parameters": {"K": 8, "r": 16, "tau": 16.0, "d": 1024, "L": 1},
    })
    return attempt, final_config, preprocess_cache


def command(task: dict[str, Any], config: Path, attempt: Path, cache: Path, device: str, gpu: int) -> list[str]:
    command = [
        sys.executable, "-u", "main.py", "--model", "hingemix", "--dataset", task["dataset"],
        "--device", device, "--seed", str(task["training_seed"]), "--data_seed", str(task["data_seed"]),
        "--batch_size", str(task["training"]["batch_size"]), "--lr", str(task["training"]["lr"]),
        "--d_token", "1024", "--n_layers", "1", "--config", str(config),
        "--output_dir", str(attempt), "--preprocess-cache-dir", str(cache),
        "--hide-test-during-training",
    ]
    if device == "cuda":
        command.extend(["--gpu", str(gpu)])
    return command


def save_test_trace(task: dict[str, Any]) -> None:
    """Persist test predictions with source-split targets without changing model inputs.

    The prediction array is retained in the model's saved output scale.  The
    source target is retained in the saved split's source scale; data_config
    records the fitted label transformation used by the training process.
    """
    attempt = Path(task["attempt_dir"])
    directory = Path(task["dataset_evidence"]["dataset_dir"])
    prediction_path = attempt / "test_predictions.npy"
    target_path = directory / "y_test.npy"
    if not prediction_path.is_file() or not target_path.is_file():
        raise RuntimeError("test prediction or source test target file is missing")
    predictions = np.load(prediction_path)
    targets = np.load(target_path)
    if len(predictions) != len(targets):
        raise RuntimeError("saved prediction length does not match source test target length")
    source_indices = directory / "idx_test.npy"
    if source_indices.is_file():
        sample_id = np.load(source_indices)
        sample_id_source = str(source_indices.resolve())
    else:
        sample_id = np.arange(len(targets), dtype=np.int64)
        sample_id_source = "generated_test_split_position"
    if len(sample_id) != len(targets):
        raise RuntimeError("source test index length does not match source test target length")
    np.savez(
        attempt / "test_prediction_trace.npz",
        y_pred_model_output_scale=predictions,
        y_true_saved_test_split_scale=targets,
        sample_id=sample_id,
        prediction_source=np.asarray(str(prediction_path.resolve())),
        target_source=np.asarray(str(target_path.resolve())),
        sample_id_source=np.asarray(sample_id_source),
        data_config_source=np.asarray(str((attempt / "data_config.yaml").resolve())),
    )


def save_diagnostics(task: dict[str, Any]) -> None:
    attempt = Path(task["attempt_dir"])
    cache_files = list((attempt / "breakpoint_cache").rglob("breakpoints_*.json"))
    if len(cache_files) != 1:
        raise RuntimeError(f"Expected one breakpoint cache, found {len(cache_files)}")
    payload = read_json(cache_files[0]) or {}
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("breakpoint cache did not save diagnostics")
    final_breakpoints = read_json(attempt / "breakpoint_final.json")
    if not isinstance(final_breakpoints, dict):
        raise RuntimeError("validation-selected checkpoint breakpoint record is missing")
    diagnostics.update({
        "dataset": task["dataset"], "strategy": task["strategy"], "training_seed": task["training_seed"],
        "config_fingerprint": task["config_fingerprint"], "cache_file": str(cache_files[0].resolve()),
        "feature_names": task["dataset_evidence"]["feature_names"],
        "feature_name_source": task["dataset_evidence"]["feature_name_source"],
        "final_checkpoint_breakpoints": final_breakpoints,
    })
    atomic_json(attempt / "breakpoint_diagnostics.json", diagnostics)


def collect_result(task: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(task["attempt_dir"])
    prediction, history = read_json(attempt / "prediction.json") or {}, read_json(attempt / "results.json") or {}
    diagnostics = read_json(attempt / "breakpoint_diagnostics.json") or {}
    metrics = prediction.get("metrics") or {}
    return {
        "config_fingerprint": task["config_fingerprint"], "attempt_dir": task["attempt_dir"],
        "model": "hingemix", "dataset": task["dataset"], "strategy": task["strategy"],
        "training_seed": task["training_seed"], "data_seed": task["data_seed"], "smoke": task["smoke"],
        "effective_parameters": (read_json(attempt / "task.json") or {}).get("effective_parameters"),
        "best_epoch": (history.get("val") or {}).get("best_epoch"),
        "validation_rmse": (history.get("val") or {}).get("best_metric"),
        "test_rmse": metrics.get("rmse"), "test_mae": metrics.get("mae"), "test_r2": metrics.get("r2"),
        "breakpoint_diagnostics": str((attempt / "breakpoint_diagnostics.json").resolve()),
        "test_prediction_trace": str((attempt / "test_prediction_trace.npz").resolve()),
        "started_at": task.get("started_at"), "finished_at": task.get("finished_at"),
    }


def check_gpus(gpus: list[int], minimum: int) -> None:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if query.returncode:
        raise SystemExit(f"nvidia-smi failed: {query.stderr.strip()}")
    available = {int(line.split(",")[0]): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    unavailable = [gpu for gpu in gpus if available.get(gpu, 0) < minimum]
    if unavailable:
        raise SystemExit(f"Refusing to start; GPUs below {minimum} MiB free: {unavailable}")
    print(f"GPU free memory (MiB): {available}")


def main() -> None:
    args = parse_args()
    if set(args.datasets) - set(DATASETS) or set(args.strategies) - set(STRATEGIES) or set(args.seeds) - set(SEEDS):
        raise SystemExit("Unknown dataset, strategy, or seed")
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("Duplicate GPU selection is not allowed")
    repo = Path(__file__).resolve().parents[1]
    if args.smoke and args.output_root == "results/hingemix_breakpoint_ablation":
        args.output_root = "results/hingemix_breakpoint_ablation_smoke"
    root = Path(args.output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) or build_manifest(repo, args)
    if manifest.get("experiment") != "hingemix_breakpoint_ablation" or manifest.get("smoke") != bool(args.smoke):
        raise SystemExit("Existing manifest belongs to another experiment mode")
    selected = [task for task in manifest["tasks"] if task["dataset"] in args.datasets and task["strategy"] in args.strategies and task["training_seed"] in args.seeds]
    if args.smoke:
        selected = manifest["tasks"]
    current_code = source_fingerprint(repo)
    current_data = {dataset: dataset_evidence(repo, dataset) for dataset in {task["dataset"] for task in selected}}
    pending: list[dict[str, Any]] = []
    states: dict[str, int] = {}
    for task in selected:
        root_dir = task_root(root, task)
        if task.get("source_fingerprint") != current_code:
            task.update(status="needs_review", failure_reason="source fingerprint differs from manifest")
        elif task.get("dataset_evidence") != current_data[task["dataset"]]:
            task.update(status="needs_review", failure_reason="dataset fingerprint differs from manifest")
        else:
            valid, reason = valid_result(root_dir, task)
            if valid:
                task["status"] = "completed"
            elif (root_dir / "parameter_result.json").exists():
                task.update(status="needs_review", failure_reason=reason)
            elif (root_dir / ".task.lock").exists() and lock_is_active(root_dir / ".task.lock"):
                task["status"] = "running"
            elif task.get("status") == "failed" and not args.retry_failed:
                pass
            else:
                task["status"] = "pending"
                pending.append(task)
        states[task["status"]] = states.get(task["status"], 0) + 1
    atomic_json(manifest_path, manifest)
    print(f"design: 3 strategies x 7 datasets x 5 seeds = 105 formal tasks; selected={len(selected)}")
    print("states:", states, "pending:", len(pending), "device:", args.device)
    for task in pending:
        print("pending:", task["strategy"], task["dataset"], f"seed={task['training_seed']}", task_root(root, task))
    if args.dry_run:
        return
    if args.device == "cuda":
        check_gpus(args.gpus, args.min_free_mib)
    workers = [0] if args.device == "cpu" else args.gpus
    children: dict[int, Any] = {}
    stopping = False

    def stop(*_unused) -> None:
        nonlocal stopping
        stopping = True
        for process, *_ in children.values():
            process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while (pending and not stopping) or children:
        busy = {item[3] for item in children.values()}
        for gpu in [item for item in workers if item not in busy]:
            if not pending or stopping:
                break
            task = pending.pop(0)
            root_dir = task_root(root, task)
            held = acquire_lock(root_dir)
            if held is None:
                task.update(status="needs_review", failure_reason="active task lock exists")
                continue
            attempt, config, cache = materialize(repo, root_dir, task)
            task.update(status="running", gpu=gpu if args.device == "cuda" else None, started_at=time.time(), attempt_dir=str(attempt.resolve()))
            atomic_json(manifest_path, manifest)
            log = (attempt / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command(task, config, attempt, cache, args.device, gpu), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, task, held, gpu, log)
        for pid, (process, task, held, gpu, log) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            release_lock(held)
            children.pop(pid)
            task.update(finished_at=time.time(), return_code=code)
            try:
                if code == 0:
                    save_test_trace(task)
                    save_diagnostics(task)
                    result = collect_result(task)
                else:
                    result = None
            except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
                result = None
                task["result_parse_error"] = str(error)
            if all(isinstance((result or {}).get(key), (int, float)) and np.isfinite(result[key]) for key in ("test_rmse", "test_mae", "test_r2")):
                atomic_json(task_root(root, task) / "parameter_result.json", result)
                task.update(status="completed", failure_reason=None)
            else:
                task.update(status="failed", failure_reason=task.get("result_parse_error", f"main.py exited with {code}"))
            atomic_json(manifest_path, manifest)
            print(f"{task['status']}: gpu={gpu} {task['strategy']} {task['dataset']} seed={task['training_seed']}", flush=True)
        if children:
            time.sleep(0.5)
    if stopping:
        raise SystemExit("Interrupted: child processes started by this runner were terminated")


if __name__ == "__main__":
    main()
