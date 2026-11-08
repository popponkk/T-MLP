#!/usr/bin/env python3
"""Run resumable HingeMix training-set-size experiments without changing saved splits.

Each percentage is a nested, deterministic subset of the saved training split.
Validation and test arrays are never subsetted.  The runner materializes one
configuration, preprocessing cache, breakpoint cache, log, and checkpoint tree
per dataset/percentage/training-seed task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
PERCENTAGES = (5, 10, 20, 40, 60, 80, 100)
SEEDS = (0, 1, 2, 3, 4)
FINAL = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def resolve_dataset(repo: Path, dataset: str) -> Path:
    # Match data/env.py priority: built-in discovered datasets precede custom data.
    candidates = [
        repo / "data" / "datasets" / dataset,
        repo / "data" / "custom_datasets" / dataset,
    ]
    found = [path for path in candidates if (path / "info.json").is_file()]
    if not found:
        raise FileNotFoundError(f"No registered dataset directory found for {dataset}: {candidates}")
    if len(found) > 1:
        print(f"warning: {dataset} exists in multiple roots; using {found[0]}", flush=True)
    return found[0]


def dataset_evidence(repo: Path, dataset: str) -> dict[str, Any]:
    directory = resolve_dataset(repo, dataset)
    required = [directory / "info.json", directory / "X_num_train.npy", directory / "y_train.npy"]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{dataset} lacks required files: {absent}")
    train_size = int(np.load(directory / "y_train.npy", mmap_mode="r").shape[0])
    source_index = directory / "idx_train.npy"
    if source_index.is_file():
        source_values = np.load(source_index, mmap_mode="r")
        if source_values.ndim != 1 or len(source_values) != train_size:
            raise ValueError(f"{source_index} is incompatible with the saved training split")
        source_index_sha256 = sha256_file(source_index)
    else:
        source_index_sha256 = None
    files = sorted(directory.glob("*.npy")) + [directory / "info.json"]
    return {
        "dataset_dir": str(directory.resolve()),
        "train_size": train_size,
        "source_train_index_path": str(source_index.resolve()) if source_index.is_file() else None,
        "source_train_index_sha256": source_index_sha256,
        "dataset_files_sha256": {
            path.name: sha256_file(path) for path in files if path.is_file()
        },
    }


def subset_positions(train_size: int, percentage: int, subset_seed: int) -> np.ndarray:
    count = math.floor(train_size * percentage / 100.0)
    if count <= 0:
        raise ValueError(f"{percentage}% of {train_size} training rows selects no samples")
    # One permutation per dataset makes all percentages nested by construction.
    order = np.random.default_rng(subset_seed).permutation(train_size)
    return np.sort(order[:count].astype(np.int64, copy=False))


def source_fingerprint(repo: Path) -> dict[str, str]:
    paths = (
        repo / "models" / "hingemix.py",
        repo / "models" / "ggpl_tmlp.py",
        repo / "data" / "processor.py",
        repo / "main.py",
        repo / "configs" / "default" / "hingemix.yaml",
        repo / "scripts" / "run_hingemix_sample_size.py",
    )
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--fractions", nargs="+", type=int, default=list(PERCENTAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", default="results/hingemix_sample_size")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    return parser.parse_args()


def task_root(root: Path, task: dict[str, Any]) -> Path:
    return (
        root
        / task["dataset"]
        / f"fraction_{task['subset']['percentage']:03d}pct"
        / f"seed_{task['training_seed']}"
    )


def build_manifest(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True
    ).stdout.strip() or None
    code = source_fingerprint(repo)
    tasks: list[dict[str, Any]] = []
    for dataset in DATASETS:
        evidence = dataset_evidence(repo, dataset)
        for percentage in PERCENTAGES:
            positions = subset_positions(evidence["train_size"], percentage, args.subset_seed)
            subset = {
                "selection_rule": "nested_rng_permutation_then_floor",
                "subset_seed": args.subset_seed,
                "percentage": percentage,
                "source_train_size": evidence["train_size"],
                "selected_train_size": int(len(positions)),
                "train_positions_sha256": sha256_bytes(positions.tobytes()),
                "source_train_index_sha256": evidence["source_train_index_sha256"],
            }
            for training_seed in SEEDS:
                identity = {
                    "model": "hingemix",
                    "dataset": dataset,
                    "training_seed": training_seed,
                    "data_seed": args.data_seed,
                    "parameters": FINAL,
                    "training": {"lr": args.lr, "batch_size": args.batch_size},
                    "subset": subset,
                    "dataset_evidence": evidence,
                    "source_fingerprint": code,
                    "git_revision": revision,
                }
                tasks.append({
                    **identity,
                    "config_fingerprint": fingerprint(identity),
                    "status": "pending",
                })
    expected = len(DATASETS) * len(PERCENTAGES) * len(SEEDS)
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} tasks, got {len(tasks)}")
    return {
        "schema_version": 1,
        "experiment": "hingemix_sample_size",
        "created_at": time.time(),
        "expected_tasks": expected,
        "datasets": list(DATASETS),
        "percentages": list(PERCENTAGES),
        "training_seeds": list(SEEDS),
        "data_seed": args.data_seed,
        "subset_seed": args.subset_seed,
        "final_parameters": FINAL,
        "tasks": tasks,
    }


def valid_result(root: Path, task: dict[str, Any]) -> tuple[bool, str | None]:
    marker = read_json(root / "parameter_result.json")
    if not marker or marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "missing or conflicting completed result"
    attempt_dir = marker.get("attempt_dir")
    if not isinstance(attempt_dir, str):
        return False, "completed result has no attempt directory"
    attempt = Path(attempt_dir)
    metadata = read_json(attempt / "task.json")
    prediction = read_json(attempt / "prediction.json")
    if not metadata or metadata.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt metadata fingerprint mismatch"
    if not isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
        return False, "test RMSE is missing or invalid"
    return True, None


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name.rsplit("_", 1)[1])
        for path in attempts.glob("attempt_*")
        if path.name.rsplit("_", 1)[-1].isdigit()
    ]
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


def materialize(repo: Path, root: Path, task: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    attempt = next_attempt(root)
    attempt.mkdir(parents=True)
    cache_dir = attempt / "preprocessing_cache"
    cache_dir.mkdir()
    positions = subset_positions(
        task["subset"]["source_train_size"], task["subset"]["percentage"], task["subset"]["subset_seed"]
    )
    if sha256_bytes(positions.tobytes()) != task["subset"]["train_positions_sha256"]:
        raise RuntimeError("regenerated training subset does not match the manifest")
    positions_path = cache_dir / "train_subset_positions.npy"
    np.save(positions_path, positions)
    # transform_dataset includes info.json and .npy files in its cache signature.
    atomic_json(cache_dir / "info.json", {
        "experiment": "hingemix_sample_size",
        "dataset": task["dataset"],
        "subset": task["subset"],
        "dataset_evidence": task["dataset_evidence"],
        "config_fingerprint": task["config_fingerprint"],
    })

    config = yaml.safe_load((repo / "configs" / "default" / "hingemix.yaml").read_text(encoding="utf-8"))
    config["model"].update(FINAL)
    config["model"]["model_name"] = "hingemix"
    config["model"]["breakpoint_cache_dir"] = str(attempt / "breakpoint_cache")
    config["training"].update(lr=task["training"]["lr"], batch_size=task["training"]["batch_size"])
    if any(config["model"].get(key) != value for key, value in FINAL.items()):
        raise RuntimeError("HingeMix final parameters were not materialized")
    config_path = attempt / "final_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(attempt / "task.json", {
        **task,
        "attempt_dir": str(attempt.resolve()),
        "preprocessing_cache_dir": str(cache_dir.resolve()),
        "train_subset_indices_path": str(positions_path.resolve()),
        "effective_parameters": {"K": 8, "r": 16, "tau": 16.0, "d": 1024, "L": 1},
    })
    return attempt, config_path, positions_path, cache_dir


def command(task: dict[str, Any], config: Path, attempt: Path, positions: Path, cache_dir: Path, gpu: int) -> list[str]:
    return [
        sys.executable, "-u", "main.py",
        "--model", "hingemix", "--dataset", task["dataset"],
        "--device", "cuda", "--gpu", str(gpu),
        "--seed", str(task["training_seed"]), "--data_seed", str(task["data_seed"]),
        "--batch_size", str(task["training"]["batch_size"]), "--lr", str(task["training"]["lr"]),
        "--d_token", "1024", "--n_layers", "1",
        "--config", str(config), "--output_dir", str(attempt),
        "--train-subset-indices", str(positions),
        "--preprocess-cache-dir", str(cache_dir),
    ]


def collect_result(task: dict[str, Any]) -> dict[str, Any]:
    attempt = Path(task["attempt_dir"])
    prediction = read_json(attempt / "prediction.json") or {}
    history = read_json(attempt / "results.json") or {}
    log = (attempt / "training.log").read_text(encoding="utf-8", errors="replace")
    parameters = re.search(r"\[hingemix\] parameters=(\d+)\s+trainable_parameters=(\d+)", log)
    return {
        "config_fingerprint": task["config_fingerprint"],
        "attempt_dir": task["attempt_dir"],
        "model": "hingemix",
        "dataset": task["dataset"],
        "training_seed": task["training_seed"],
        "data_seed": task["data_seed"],
        "subset": task["subset"],
        "effective_parameters": (read_json(attempt / "task.json") or {}).get("effective_parameters"),
        "final_config": yaml.safe_load((attempt / "final_config.yaml").read_text(encoding="utf-8")),
        "validation_rmse": (history.get("val") or {}).get("best_metric"),
        "best_epoch": (history.get("val") or {}).get("best_epoch"),
        "test_rmse": (prediction.get("metrics") or {}).get("rmse"),
        "parameter_count": int(parameters.group(1)) if parameters else None,
        "trainable_parameter_count": int(parameters.group(2)) if parameters else None,
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "training_wall_seconds": task.get("finished_at", 0) - task.get("started_at", 0),
    }


def check_gpus(gpus: list[int], minimum: int) -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if query.returncode:
        raise SystemExit(f"nvidia-smi failed: {query.stderr.strip()}")
    available = {int(line.split(",")[0]): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    print(f"GPU free memory (MiB): {available}")
    unavailable = [gpu for gpu in gpus if available.get(gpu, 0) < minimum]
    if unavailable:
        raise SystemExit(f"Refusing to start; GPUs below {minimum} MiB free: {unavailable}")


def main() -> None:
    args = parse_args()
    invalid = (
        set(args.datasets) - set(DATASETS)
        or set(args.fractions) - set(PERCENTAGES)
        or set(args.seeds) - set(SEEDS)
        or len(set(args.gpus)) != len(args.gpus)
    )
    if invalid:
        raise SystemExit("Unknown dataset/fraction/seed or duplicate GPU selection")
    repo, root = Path(__file__).resolve().parents[1], Path(args.output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) or build_manifest(repo, args)
    expected = len(DATASETS) * len(PERCENTAGES) * len(SEEDS)
    if (
        manifest.get("expected_tasks") != expected
        or manifest.get("final_parameters") != FINAL
        or manifest.get("data_seed") != args.data_seed
        or manifest.get("subset_seed") != args.subset_seed
        or any(task.get("training", {}).get("lr") != args.lr for task in manifest.get("tasks", []))
        or any(task.get("training", {}).get("batch_size") != args.batch_size for task in manifest.get("tasks", []))
    ):
        raise SystemExit("Existing manifest conflicts with the fixed HingeMix sample-size design")
    selected = [
        task for task in manifest["tasks"]
        if task["dataset"] in args.datasets
        and task["subset"]["percentage"] in args.fractions
        and task["training_seed"] in args.seeds
    ]
    current_code = source_fingerprint(repo)
    current_data = {dataset: dataset_evidence(repo, dataset) for dataset in args.datasets}
    states: dict[str, int] = {}
    pending: list[dict[str, Any]] = []
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
    print(f"design: 1 HingeMix model x {len(DATASETS)} datasets x {len(PERCENTAGES)} fractions x {len(SEEDS)} seeds = {expected}")
    print("parameters:", FINAL, "lr=", args.lr, "batch_size=", args.batch_size,
          "data_seed=", args.data_seed, "subset_seed=", args.subset_seed)
    print("states:", states, "pending:", len(pending))
    for task in pending:
        print("pending:", task["dataset"], f"fraction={task['subset']['percentage']}%",
              f"n={task['subset']['selected_train_size']}", f"seed={task['training_seed']}", task_root(root, task))
    if args.dry_run:
        return
    check_gpus(args.gpus, args.min_free_mib)
    children: dict[int, Any] = {}
    stopping = False

    def stop(*_unused) -> None:
        nonlocal stopping
        stopping = True
        for process, _, _, _ in children.values():
            process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while (pending and not stopping) or children:
        busy = {entry[3] for entry in children.values()}
        free = [gpu for gpu in args.gpus if gpu not in busy]
        while pending and free and not stopping:
            task = pending.pop(0)
            root_dir = task_root(root, task)
            held = acquire_lock(root_dir)
            if held is None:
                task.update(status="needs_review", failure_reason="task lock exists")
                atomic_json(manifest_path, manifest)
                continue
            gpu = free.pop(0)
            attempt, config, positions, cache_dir = materialize(repo, root_dir, task)
            task.update(status="running", gpu=gpu, started_at=time.time(), attempt_dir=str(attempt.resolve()))
            atomic_json(manifest_path, manifest)
            log = (attempt / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command(task, config, attempt, positions, cache_dir, gpu),
                cwd=repo, stdout=log, stderr=subprocess.STDOUT,
            )
            children[process.pid] = (process, (task, log), held, gpu)
        for pid, (process, (task, log), held, gpu) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            release_lock(held)
            children.pop(pid)
            task.update(finished_at=time.time(), return_code=code)
            try:
                result = collect_result(task) if code == 0 else None
            except (OSError, ValueError, yaml.YAMLError) as error:
                result = None
                task["result_parse_error"] = str(error)
            if isinstance((result or {}).get("test_rmse"), (int, float)):
                result["completed_at"] = time.time()
                atomic_json(task_root(root, task) / "parameter_result.json", result)
                task.update(status="completed", failure_reason=None)
            else:
                task.update(
                    status="failed",
                    failure_reason=task.get("result_parse_error", f"main.py exited with {code}"),
                )
            atomic_json(manifest_path, manifest)
            print(
                f"{task['status']}: gpu={gpu} {task['dataset']} "
                f"fraction={task['subset']['percentage']}% seed={task['training_seed']}",
                flush=True,
            )
        if children:
            time.sleep(0.5)
    if stopping:
        raise SystemExit("Interrupted: only child processes started by this runner were terminated")


if __name__ == "__main__":
    main()
