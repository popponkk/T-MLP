#!/usr/bin/env python3
"""Run the two no-IC datasets with formal HingeMix, ablations, and baselines."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


DATASETS = ("raiderstream_no_ic", "cachesweep_no_ic")
SEEDS = (0, 1, 2, 3, 4)
FINAL = {"num_breakpoints": 8, "graph_dynamic_rank": 16, "graph_temperature": 16.0, "d_token": 1024, "n_layers": 1}
NEURAL_BASELINES = ("excel-former", "ft-transformer", "node", "mlp", "autoint", "tabm")
TREE_BASELINES = ("lightgbm", "xgboost", "catboost")
SPECS = {
    "full": {"model": "hingemix", "ablation": None, "kind": "hingemix", "display": "HingeMix"},
    "no_graph": {"model": "hingemix_ablation", "ablation": "no_graph", "kind": "hingemix", "display": "HingeMix-no_graph"},
    "linear": {"model": "hingemix_ablation", "ablation": "linear", "kind": "hingemix", "display": "HingeMix-linear"},
    "linear_no_graph": {"model": "hingemix_ablation", "ablation": "linear_no_graph", "kind": "hingemix", "display": "HingeMix-linear_no_graph"},
    **{name: {"model": name, "ablation": None, "kind": "neural", "display": name} for name in NEURAL_BASELINES},
    **{name: {"model": name, "ablation": None, "kind": "tree", "display": name} for name in TREE_BASELINES},
}


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_fingerprint(repo: Path, dataset: str) -> dict:
    for root in (repo / "data" / "datasets", repo / "data" / "custom_datasets"):
        directory = root / dataset
        if directory.is_dir():
            files = sorted(directory.glob("*.npy")) + [directory / "info.json", directory / "provenance.json"]
            return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}
    raise FileNotFoundError(f"No dataset directory for {dataset}")


def source_fingerprint(repo: Path) -> dict:
    paths = (repo / "models" / "hingemix.py", repo / "models" / "hingemix_ablation.py", repo / "main.py")
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def build_manifest(repo: Path, args) -> dict:
    tasks = []
    for experiment, spec in SPECS.items():
        for dataset in args.datasets:
            for seed in args.seeds:
                identity = {"experiment": experiment, "spec": spec, "dataset": dataset, "seed": seed, "data_seed": args.data_seed,
                            "parameters": FINAL if spec["kind"] == "hingemix" else {}, "lr": args.lr if spec["kind"] in {"hingemix", "neural"} else None,
                            "batch_size": args.batch_size if spec["kind"] in {"hingemix", "neural"} else None,
                            "dataset_fingerprint": data_fingerprint(repo, dataset), "source_fingerprint": source_fingerprint(repo)}
                tasks.append({**identity, "config_fingerprint": fingerprint(identity), "status": "pending"})
    if len(tasks) != 130:
        raise AssertionError(f"Expected 130 tasks, got {len(tasks)}")
    return {"schema_version": 1, "experiment": "hingemix_no_ic", "created_at": time.time(), "physical_training_tasks": len(tasks),
            "logical_full_ablation_reference": "full references the same HingeMix task; it is not a second training task.", "tasks": tasks}


def task_root(root: Path, task: dict) -> Path:
    return root / task["experiment"] / task["dataset"] / f"seed_{task['seed']}"


def valid(task: dict, root: Path) -> tuple[bool, str | None]:
    marker = read(root / "parameter_result.json")
    if not marker or marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "missing or conflicting result marker"
    attempt = Path(marker.get("attempt_dir", ""))
    meta, prediction = read(attempt / "task.json"), read(attempt / "prediction.json")
    if not meta or meta.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt metadata fingerprint mismatch"
    if not isinstance((prediction or {}).get("metrics", {}).get("rmse"), (int, float)):
        return False, "valid raw-scale test RMSE missing"
    return True, None


def next_attempt(root: Path) -> Path:
    attempts = root / "attempts"; attempts.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.rsplit("_", 1)[1]) for path in attempts.glob("attempt_*") if path.name.rsplit("_", 1)[-1].isdigit()]
    return attempts / f"attempt_{max(numbers, default=0) + 1:03d}"


def lock(root: Path):
    root.mkdir(parents=True, exist_ok=True); path = root / ".task.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(handle, str(os.getpid()).encode()); return path, handle


def unlock(item) -> None:
    if item:
        path, handle = item; os.close(handle); path.unlink(missing_ok=True)


def materialize(repo: Path, root: Path, task: dict) -> tuple[Path, Path]:
    spec = task["spec"]
    config_name = "hingemix.yaml" if spec["model"] == "hingemix" else "hingemix_ablation.yaml" if spec["model"] == "hingemix_ablation" else f"{spec['model']}.yaml"
    config = yaml.safe_load((repo / "configs" / "default" / config_name).read_text(encoding="utf-8"))
    attempt = next_attempt(root); attempt.mkdir(parents=True)
    if spec["kind"] == "hingemix":
        config["model"].update(FINAL); config["model"]["model_name"] = spec["model"]
        config["model"]["breakpoint_cache_dir"] = str(attempt / "breakpoint_cache")
        if spec["ablation"] is not None: config["model"]["ablation"] = spec["ablation"]
    if spec["kind"] in {"hingemix", "neural"}:
        config.setdefault("training", {}).update(lr=task["lr"], batch_size=task["batch_size"])
    config_path = attempt / "final_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    effective = {"K": FINAL["num_breakpoints"] if spec["kind"] == "hingemix" and spec["ablation"] not in {"linear", "linear_no_graph"} else None,
                 "r": FINAL["graph_dynamic_rank"] if spec["kind"] == "hingemix" and spec["ablation"] not in {"no_graph", "linear_no_graph"} else None,
                 "tau": FINAL["graph_temperature"] if spec["kind"] == "hingemix" and spec["ablation"] not in {"no_graph", "linear_no_graph"} else None,
                 "d": FINAL["d_token"] if spec["kind"] == "hingemix" else None, "L": FINAL["n_layers"] if spec["kind"] == "hingemix" else None}
    atomic(attempt / "task.json", {**task, "attempt_dir": str(attempt), "effective_parameters": effective,
                                    "test_target_path": str(next((repo / "data" / part / task["dataset"] / "y_test.npy" for part in ("datasets", "custom_datasets") if (repo / "data" / part / task["dataset"] / "y_test.npy").is_file()), "")})
    return attempt, config_path


def command(task: dict, config: Path, attempt: Path, device: str, gpu: int | None) -> list[str]:
    spec = task["spec"]
    cmd = [sys.executable, "-u", "main.py", "--model", spec["model"], "--dataset", task["dataset"], "--device", device,
           "--seed", str(task["seed"]), "--data_seed", str(task["data_seed"]), "--config", str(config), "--output_dir", str(attempt)]
    if gpu is not None: cmd.extend(("--gpu", str(gpu)))
    if spec["kind"] in {"hingemix", "neural"}: cmd.extend(("--lr", str(task["lr"]), "--batch_size", str(task["batch_size"])))
    if spec["kind"] == "hingemix": cmd.extend(("--d_token", "1024", "--n_layers", "1"))
    if spec["ablation"] is not None: cmd.extend(("--ablation", spec["ablation"]))
    return cmd


def collect(task: dict) -> dict:
    attempt = Path(task["attempt_dir"]); prediction, history = read(attempt / "prediction.json") or {}, read(attempt / "results.json") or {}
    return {"config_fingerprint": task["config_fingerprint"], "attempt_dir": str(attempt), "experiment": task["experiment"], "model": task["spec"]["model"],
            "ablation": task["spec"]["ablation"], "dataset": task["dataset"], "training_seed": task["seed"], "data_seed": task["data_seed"],
            "dataset_fingerprint": task["dataset_fingerprint"], "final_config": yaml.safe_load((attempt / "final_config.yaml").read_text(encoding="utf-8")),
            "test_rmse": (prediction.get("metrics") or {}).get("rmse"), "validation_rmse": (history.get("val") or {}).get("best_metric"),
            "best_epoch": (history.get("val") or {}).get("best_epoch"), "started_at": task.get("started_at"), "finished_at": time.time()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS)); parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1]); parser.add_argument("--output-root", default="results/hingemix_no_ic")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--retry-failed", action="store_true"); parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5); parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(); repo, root = Path(__file__).resolve().parents[1], Path(args.output_root).resolve()
    manifest_path = root / "manifest.json"; manifest = read(manifest_path) or build_manifest(repo, args)
    if manifest.get("physical_training_tasks") != 130: raise SystemExit("Existing manifest conflicts with the fixed no-IC design")
    selected = [task for task in manifest["tasks"] if task["dataset"] in args.datasets and task["experiment"] in args.models and task["seed"] in args.seeds]
    pending, states = [], {}
    for task in selected:
        good, reason = valid(task, task_root(root, task))
        if good: task["status"] = "completed"
        elif (task_root(root, task) / "parameter_result.json").exists(): task.update(status="needs_review", failure_reason=reason)
        elif task.get("status") == "failed" and not args.retry_failed: pass
        else: task["status"] = "pending"; pending.append(task)
        states[task["status"]] = states.get(task["status"], 0) + 1
    atomic(manifest_path, manifest); print(f"no-IC design: {len(SPECS)} physical models x 2 datasets x 5 seeds = 130 training tasks; selected={len(selected)}")
    print("states:", states, "pending:", len(pending)); print("full ablation is a reference to each full HingeMix run, not an extra training task")
    if args.dry_run: return
    check = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if check.returncode:
        raise SystemExit(f"nvidia-smi failed; refusing to launch GPU tasks: {check.stderr.strip()}")
    memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in check.stdout.splitlines()}
    print("GPU free memory (MiB):", memory)
    if any(memory.get(gpu, 0) < 2048 for gpu in args.gpus):
        raise SystemExit("At least one requested GPU has less than 2048 MiB free; refusing to contend with existing work")
    children, stopping = {}, False
    def stop(*_):
        nonlocal stopping; stopping = True
        for process, _, _, _ in children.values(): process.terminate()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    while (pending and not stopping) or children:
        used_gpu = {entry[3] for entry in children.values() if entry[3] is not None}; cpu_busy = any(entry[3] is None for entry in children.values())
        slots = [("cuda", gpu) for gpu in args.gpus if gpu not in used_gpu] + ([] if cpu_busy else [("cpu", None)])
        while pending and slots and not stopping:
            choice = next(((task_index, slot_index) for slot_index, slot in enumerate(slots) for task_index, task in enumerate(pending)
                           if (task["spec"]["kind"] == "tree") == (slot[0] == "cpu")), None)
            if choice is None: break
            task_index, slot_index = choice
            task = pending.pop(task_index); device, gpu = slots.pop(slot_index); root_dir = task_root(root, task); held = lock(root_dir)
            if held is None: task.update(status="needs_review", failure_reason="task lock exists"); continue
            attempt, config = materialize(repo, root_dir, task); task.update(status="running", attempt_dir=str(attempt), started_at=time.time(), device=device, gpu=gpu); atomic(manifest_path, manifest)
            log = (attempt / "training.log").open("w", encoding="utf-8"); process = subprocess.Popen(command(task, config, attempt, device, gpu), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, (task, log), held, gpu)
        for pid, (process, (task, log), held, gpu) in list(children.items()):
            code = process.poll()
            if code is None: continue
            log.close(); unlock(held); children.pop(pid); task["finished_at"] = time.time(); result = collect(task) if code == 0 else None
            if isinstance((result or {}).get("test_rmse"), (int, float)): atomic(task_root(root, task) / "parameter_result.json", result); task.update(status="completed", failure_reason=None)
            else: task.update(status="failed", failure_reason=f"main.py exited {code}" if code else "test RMSE missing")
            atomic(manifest_path, manifest); print(f"{task['status']}: device={task['device']} {task['experiment']} {task['dataset']} seed={task['seed']}", flush=True)
        if children: time.sleep(0.5)
    if stopping: raise SystemExit("Interrupted only child processes started by this runner")


if __name__ == "__main__": main()
