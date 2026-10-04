"""Run the four formal HingeMix tau=16 ablations in an isolated namespace."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


DATASETS = ("hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161", "raiderstream", "stream", "cachesweep")
SEEDS = (0, 1, 2, 3, 4)
FINAL = {"num_breakpoints": 8, "graph_dynamic_rank": 16, "graph_temperature": 16.0, "d_token": 1024, "n_layers": 1}
# Historical channel-removal and framework-equivalence controls remain in the
# legacy manifest but are intentionally outside the four-method formal table.
EXPERIMENTS = {
    "full": {"model": "hingemix", "ablation": None, "tokenizer": "ggpl", "graph": True, "channel": True},
    "no_graph": {"model": "hingemix_ablation", "ablation": "no_graph", "tokenizer": "ggpl", "graph": False, "channel": True},
    "linear": {"model": "hingemix_ablation", "ablation": "linear", "tokenizer": "shared_linear", "graph": True, "channel": True},
    "linear_no_graph": {"model": "hingemix_ablation", "ablation": "linear_no_graph", "tokenizer": "shared_linear", "graph": False, "channel": True},
}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fingerprint(repo):
    paths = (repo / "models" / "hingemix.py", repo / "models" / "hingemix_ablation.py", repo / "main.py")
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--models", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--output-root", default="results/hingemix_ablation_final_tau16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-free-mib", type=int, default=2048)
    return p.parse_args()


def build_manifest(repo, args):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True).stdout.strip() or None
    tasks = []
    for name, spec in EXPERIMENTS.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                identity = {"experiment": name, "spec": spec, "dataset": dataset, "seed": seed,
                            "parameters": FINAL, "data_seed": args.data_seed, "lr": args.lr,
                            "batch_size": args.batch_size, "code": source_fingerprint(repo), "revision": revision}
                tasks.append({**identity, "config_fingerprint": digest(identity), "status": "pending"})
    if len(tasks) != 140:
        raise AssertionError(f"Expected 140 tasks, got {len(tasks)}")
    return {"schema_version": 1, "experiment": "hingemix_ablation_final_tau16", "created_at": time.time(),
            "final_parameters": FINAL, "expected_tasks": 140, "tasks": tasks}


def task_root(root, task):
    return root / task["experiment"] / task["dataset"] / f"seed_{task['seed']}"


def valid_result(root, task):
    marker = read_json(root / "parameter_result.json")
    if not marker or marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "missing or conflicting completed result"
    attempt = marker.get("attempt_dir")
    if not isinstance(attempt, str):
        return False, "completed result has no attempt directory"
    attempt = Path(attempt)
    meta, prediction = read_json(attempt / "task.json"), read_json(attempt / "prediction.json")
    if not meta or meta.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt metadata fingerprint mismatch"
    if not isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
        return False, "missing test RMSE"
    return True, None


def next_attempt(root):
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.rsplit("_", 1)[1]) for p in attempts.glob("attempt_*") if p.name.rsplit("_", 1)[-1].isdigit()]
    return attempts / f"attempt_{max(numbers, default=0) + 1:03d}"


def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".task.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if lock_is_active(path):
            return None
        # A dead scheduler must not permanently prevent a safe resume.
        path.unlink(missing_ok=True)
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    return path, handle


def unlock(item):
    if item:
        path, handle = item
        os.close(handle)
        path.unlink(missing_ok=True)


def lock_is_active(path):
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, ValueError):
        # A malformed or inaccessible lock is treated as active for safety.
        return True
    return True


def materialize(repo, root, task):
    spec = task["spec"]
    config_name = "hingemix.yaml" if spec["model"] == "hingemix" else "hingemix_ablation.yaml"
    config = yaml.safe_load((repo / "configs" / "default" / config_name).read_text(encoding="utf-8"))
    attempt = next_attempt(root)
    attempt.mkdir(parents=True)
    config["model"].update(FINAL)
    config["model"]["model_name"] = spec["model"]
    if spec["ablation"] is not None:
        config["model"]["ablation"] = spec["ablation"]
    config["model"]["breakpoint_cache_dir"] = str(attempt / "breakpoint_cache")
    config["training"].update(lr=task["lr"], batch_size=task["batch_size"])
    if any(config["model"].get(k) != v for k, v in FINAL.items()):
        raise RuntimeError(f"Final parameters were not materialized for {task['experiment']}")
    path = attempt / "final_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    effective = {"K": FINAL["num_breakpoints"] if spec["tokenizer"] == "ggpl" else None,
                 "r": FINAL["graph_dynamic_rank"] if spec["graph"] else None,
                 "tau": FINAL["graph_temperature"] if spec["graph"] else None,
                 "d": FINAL["d_token"], "L": FINAL["n_layers"]}
    atomic_json(attempt / "task.json", {**task, "effective_parameters": effective, "attempt_dir": str(attempt)})
    return attempt, path


def command(task, config, attempt, gpu):
    spec = task["spec"]
    cmd = [sys.executable, "-u", "main.py", "--model", spec["model"], "--dataset", task["dataset"],
           "--device", "cuda", "--gpu", str(gpu), "--seed", str(task["seed"]), "--data_seed", str(task["data_seed"]),
           "--batch_size", str(task["batch_size"]), "--lr", str(task["lr"]), "--d_token", "1024", "--n_layers", "1",
           "--config", str(config), "--output_dir", str(attempt)]
    if spec["ablation"] is not None:
        cmd.extend(("--ablation", spec["ablation"]))
    return cmd


def collect_result(task):
    attempt = Path(task["attempt_dir"])
    prediction = read_json(attempt / "prediction.json") or {}
    history = read_json(attempt / "results.json") or {}
    final_config = yaml.safe_load((attempt / "final_config.yaml").read_text(encoding="utf-8"))
    log = (attempt / "training.log").read_text(encoding="utf-8", errors="replace")
    task_meta = read_json(attempt / "task.json") or {}
    match = re.search(r"parameters=(\d+)(?:\s+trainable_parameters=(\d+))?", log)
    return {
        "config_fingerprint": task["config_fingerprint"],
        "attempt_dir": task["attempt_dir"],
        "model": task["spec"]["model"],
        "ablation": task["spec"]["ablation"],
        "effective_parameters": task_meta.get("effective_parameters"),
        "final_config": final_config,
        "validation_rmse": (history.get("val") or {}).get("best_metric"),
        "best_epoch": (history.get("val") or {}).get("best_epoch"),
        "test_rmse": (prediction.get("metrics") or {}).get("rmse"),
        "parameter_count": int(match.group(1)) if match else None,
        "trainable_parameter_count": int(match.group(2)) if match and match.group(2) else None,
        "gpu": task.get("gpu"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "training_wall_seconds": task.get("finished_at", 0) - task.get("started_at", 0),
    }


def check_gpus(gpus, minimum):
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if query.returncode:
        raise SystemExit(f"nvidia-smi failed: {query.stderr.strip()}")
    available = {int(line.split(",")[0]): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    print(f"GPU free memory (MiB): {available}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if processes.returncode == 0:
        text = processes.stdout.strip()
        print("Existing GPU compute processes:", text if text else "none")
    unavailable = [gpu for gpu in gpus if available.get(gpu, 0) < minimum]
    if unavailable:
        raise SystemExit(f"Refusing to start: GPUs below {minimum} MiB free: {unavailable}; observed={available}")


def main():
    args = parse_args()
    unknown = set(args.models) - set(EXPERIMENTS) or set(args.datasets) - set(DATASETS) or set(args.seeds) - set(SEEDS)
    if unknown:
        raise SystemExit(f"Unknown model, dataset, or seed selection: {unknown}")
    repo, root = Path(__file__).resolve().parents[1], Path(args.output_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) or build_manifest(repo, args)
    if len(manifest.get("tasks", [])) != 140 or manifest.get("final_parameters") != FINAL:
        raise SystemExit("Existing final-ablation manifest conflicts with the fixed 140-task HingeMix tau=16 design")
    selected = [t for t in manifest["tasks"] if t["experiment"] in args.models and t["dataset"] in args.datasets and t["seed"] in args.seeds]
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
    print(f"final design: {len(EXPERIMENTS)} models x {len(DATASETS)} datasets x {len(SEEDS)} seeds = 140")
    print("final parameters:", FINAL, "lr=", args.lr, "batch_size=", args.batch_size, "data_seed=", args.data_seed)
    print("states:", states, "pending:", len(pending))
    for task in pending:
        print("pending:", task["experiment"], task["dataset"], f"seed={task['seed']}", task_root(root, task))
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
        busy_gpus = {entry[3] for entry in children.values()}
        free_gpus = [gpu for gpu in args.gpus if gpu not in busy_gpus]
        while pending and free_gpus and not stopping:
            task = pending.pop(0); root_dir = task_root(root, task); held = lock(root_dir)
            if held is None:
                task.update(status="needs_review", failure_reason="task lock exists"); atomic_json(manifest_path, manifest); continue
            gpu = free_gpus.pop(0)
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
            log.close(); unlock(held); children.pop(pid)
            task.update(finished_at=time.time(), return_code=code)
            try:
                result = collect_result(task) if code == 0 else None
            except (OSError, ValueError, yaml.YAMLError) as error:
                result = None
                task["result_parse_error"] = str(error)
            rmse = (result or {}).get("test_rmse")
            if isinstance(rmse, (int, float)):
                result["completed_at"] = time.time()
                atomic_json(task_root(root, task) / "parameter_result.json", result)
                task.update(status="completed", failure_reason=None)
            else:
                reason = f"main.py exited with {code}" if code else "prediction.json has no valid test RMSE"
                task.update(status="failed", failure_reason=task.get("result_parse_error", reason))
            atomic_json(manifest_path, manifest)
            print(f"{task['status']}: gpu={gpu} {task['experiment']} {task['dataset']} seed={task['seed']}", flush=True)
        if children:
            time.sleep(0.5)
    if stopping:
        raise SystemExit("Interrupted: only child processes started by this runner were terminated")


if __name__ == "__main__":
    main()
