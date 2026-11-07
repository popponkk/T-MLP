"""Run the isolated four-way HingeMix ablation v2 experiment resumably."""

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

import yaml


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
SEEDS = (0, 1, 2, 3, 4)
ABLATIONS = ("full", "token_mlp", "linear", "linear_token_mlp")
SPECS = {
    "full": {"tokenizer_type": "ggpl", "mixing_type": "graph"},
    "token_mlp": {"tokenizer_type": "ggpl", "mixing_type": "token_mlp"},
    "linear": {"tokenizer_type": "shared_linear", "mixing_type": "graph"},
    "linear_token_mlp": {"tokenizer_type": "shared_linear", "mixing_type": "token_mlp"},
}
FINAL = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
    "token_mlp_ratio": 2.0,
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_fingerprint(repo: Path) -> dict:
    paths = (
        repo / "models" / "hingemix_ablation_v2.py",
        repo / "configs" / "default" / "hingemix_ablation_v2.yaml",
        repo / "main.py",
        Path(__file__),
    )
    return {str(path.resolve().relative_to(repo.resolve())): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", default="results/hingemix_ablation_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    return parser.parse_args()


def task_root(root: Path, task: dict) -> Path:
    return root / task["ablation"] / task["dataset"] / f"seed_{task['seed']}"


def build_manifest(repo: Path, args) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True).stdout.strip() or None
    tasks = []
    for ablation in ABLATIONS:
        for dataset in DATASETS:
            for seed in SEEDS:
                identity = {
                    "model": "hingemix_ablation_v2", "ablation": ablation, "spec": SPECS[ablation],
                    "dataset": dataset, "seed": seed, "data_seed": args.data_seed, "parameters": FINAL,
                    "lr": args.lr, "batch_size": args.batch_size, "code": source_fingerprint(repo), "revision": revision,
                }
                tasks.append({**identity, "config_fingerprint": digest(identity), "status": "pending"})
    if len(tasks) != 140:
        raise AssertionError(f"Expected 140 tasks, got {len(tasks)}")
    return {"schema_version": 1, "experiment": "hingemix_ablation_v2", "created_at": time.time(),
            "expected_tasks": 140, "final_parameters": FINAL, "tasks": tasks}


def valid_result(root: Path, task: dict):
    marker = read_json(root / "parameter_result.json")
    if not marker or marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "missing or conflicting completed result"
    attempt_dir = marker.get("attempt_dir")
    if not isinstance(attempt_dir, str):
        return False, "completed result has no attempt directory"
    attempt = Path(attempt_dir)
    meta, prediction = read_json(attempt / "task.json"), read_json(attempt / "prediction.json")
    if not meta or meta.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt metadata fingerprint mismatch"
    rmse = (prediction or {}).get("metrics", {}).get("rmse")
    if not isinstance(rmse, (int, float)) or not math.isfinite(rmse):
        return False, "missing test RMSE"
    return True, None


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [int(item.name.rsplit("_", 1)[1]) for item in attempts.glob("attempt_*") if item.name.rsplit("_", 1)[-1].isdigit()]
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


def materialize(repo: Path, root: Path, task: dict):
    config = yaml.safe_load((repo / "configs" / "default" / "hingemix_ablation_v2.yaml").read_text(encoding="utf-8"))
    attempt = next_attempt(root)
    attempt.mkdir(parents=True)
    spec = task["spec"]
    config["model"].update({"model_name": "hingemix_ablation_v2", "ablation": task["ablation"],
                            "d_token": FINAL["d_token"], "n_layers": FINAL["n_layers"]})
    if spec["tokenizer_type"] == "ggpl":
        config["model"].update(num_breakpoints=FINAL["num_breakpoints"], breakpoint_cache_dir=str(attempt / "breakpoint_cache"))
    else:
        for key in ("num_breakpoints", "learnable_breakpoints", "breakpoint_init", "breakpoint_fallback", "breakpoint_cache_dir"):
            config["model"].pop(key, None)
    if spec["mixing_type"] == "graph":
        config["model"].update(graph_dynamic_rank=FINAL["graph_dynamic_rank"], graph_temperature=FINAL["graph_temperature"])
    else:
        config["model"].update(token_mlp_ratio=FINAL["token_mlp_ratio"])
        config["model"].pop("graph_dynamic_rank", None)
        config["model"].pop("graph_temperature", None)
    config["training"].update(lr=task["lr"], batch_size=task["batch_size"])
    path = attempt / "final_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    effective = {
        "K": FINAL["num_breakpoints"] if spec["tokenizer_type"] == "ggpl" else None,
        "r": FINAL["graph_dynamic_rank"] if spec["mixing_type"] == "graph" else None,
        "tau": FINAL["graph_temperature"] if spec["mixing_type"] == "graph" else None,
        "token_mlp_ratio": FINAL["token_mlp_ratio"] if spec["mixing_type"] == "token_mlp" else None,
        "d": FINAL["d_token"], "L": FINAL["n_layers"],
    }
    atomic_json(attempt / "task.json", {**task, "effective_parameters": effective, "attempt_dir": str(attempt)})
    return attempt, path


def command(task: dict, config_path: Path, attempt: Path, gpu: int) -> list[str]:
    return [sys.executable, "-u", "main.py", "--model", "hingemix_ablation_v2", "--ablation", task["ablation"],
            "--dataset", task["dataset"], "--device", "cuda", "--gpu", str(gpu), "--seed", str(task["seed"]),
            "--data_seed", str(task["data_seed"]), "--batch_size", str(task["batch_size"]), "--lr", str(task["lr"]),
            "--d_token", str(FINAL["d_token"]), "--n_layers", str(FINAL["n_layers"]),
            "--config", str(config_path), "--output_dir", str(attempt)]


def collect_result(task: dict) -> dict:
    attempt = Path(task["attempt_dir"])
    prediction, history, meta = read_json(attempt / "prediction.json") or {}, read_json(attempt / "results.json") or {}, read_json(attempt / "task.json") or {}
    log = (attempt / "training.log").read_text(encoding="utf-8", errors="replace")
    counts = re.search(r"parameters=(\d+)\s+trainable_parameters=(\d+)", log)
    branches = re.search(r"token_mlp_parameters=(\d+)\s+channel_mlp_parameters=(\d+)", log)
    return {"config_fingerprint": task["config_fingerprint"], "attempt_dir": str(attempt), "model": task["model"],
            "ablation": task["ablation"], "dataset": task["dataset"], "seed": task["seed"],
            "effective_parameters": meta.get("effective_parameters"),
            "final_config": yaml.safe_load((attempt / "final_config.yaml").read_text(encoding="utf-8")),
            "validation_rmse": (history.get("val") or {}).get("best_metric"), "best_epoch": (history.get("val") or {}).get("best_epoch"),
            "test_rmse": (prediction.get("metrics") or {}).get("rmse"),
            "parameter_count": int(counts.group(1)) if counts else None,
            "trainable_parameter_count": int(counts.group(2)) if counts else None,
            "token_mlp_parameter_count": int(branches.group(1)) if branches else None,
            "channel_mlp_parameter_count": int(branches.group(2)) if branches else None,
            "gpu": task.get("gpu"), "started_at": task.get("started_at"), "finished_at": task.get("finished_at"),
            "training_wall_seconds": task.get("finished_at", 0) - task.get("started_at", 0)}


def check_gpus(gpus: list[int], minimum: int) -> None:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if query.returncode:
        raise SystemExit(f"nvidia-smi failed: {query.stderr.strip()}")
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    print(f"GPU free memory (MiB): {free}")
    unavailable = [gpu for gpu in gpus if free.get(gpu, 0) < minimum]
    if unavailable:
        raise SystemExit(f"Refusing to start; GPUs below {minimum} MiB free: {unavailable}")


def main() -> None:
    args = parse_args()
    unknown = set(args.ablations) - set(ABLATIONS) or set(args.datasets) - set(DATASETS) or set(args.seeds) - set(SEEDS)
    if unknown:
        raise SystemExit(f"Unknown ablation, dataset, or seed selection: {unknown}")
    repo, root = Path(__file__).resolve().parents[1], Path(args.output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) or build_manifest(repo, args)
    if manifest.get("expected_tasks") != 140 or manifest.get("final_parameters") != FINAL:
        raise SystemExit("Existing manifest conflicts with the fixed HingeMix ablation v2 design")
    selected = [task for task in manifest["tasks"] if task["ablation"] in args.ablations and task["dataset"] in args.datasets and task["seed"] in args.seeds]
    states, pending = {}, []
    for task in selected:
        root_dir = task_root(root, task)
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
    print(f"design: {len(ABLATIONS)} ablations x {len(DATASETS)} datasets x {len(SEEDS)} seeds = 140")
    print("parameters:", FINAL, "lr=", args.lr, "batch_size=", args.batch_size, "data_seed=", args.data_seed)
    print("states:", states, "pending:", len(pending))
    for task in pending:
        print("pending:", task["ablation"], task["dataset"], f"seed={task['seed']}", task_root(root, task))
    if args.dry_run:
        return
    check_gpus(args.gpus, args.min_free_mib)
    children, stopping = {}, False
    def stop(*_):
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
            attempt, config = materialize(repo, root_dir, task)
            task.update(status="running", gpu=gpu, started_at=time.time(), attempt_dir=str(attempt))
            atomic_json(manifest_path, manifest)
            log = (attempt / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command(task, config, attempt, gpu), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, (task, log), held, gpu)
        for pid, (process, (task, log), held, gpu) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); release_lock(held); children.pop(pid)
            task.update(finished_at=time.time(), return_code=code)
            try:
                result = collect_result(task) if code == 0 else None
            except (OSError, ValueError, yaml.YAMLError) as error:
                result, task["result_parse_error"] = None, str(error)
            if isinstance((result or {}).get("test_rmse"), (int, float)):
                result["completed_at"] = time.time()
                atomic_json(task_root(root, task) / "parameter_result.json", result)
                task.update(status="completed", failure_reason=None)
            else:
                task.update(status="failed", failure_reason=task.get("result_parse_error", f"main.py exited with {code}"))
            atomic_json(manifest_path, manifest)
            print(f"{task['status']}: gpu={gpu} {task['ablation']} {task['dataset']} seed={task['seed']}", flush=True)
        if children:
            time.sleep(0.5)
    if stopping:
        raise SystemExit("Interrupted: only child processes started by this runner were terminated")


if __name__ == "__main__":
    main()
