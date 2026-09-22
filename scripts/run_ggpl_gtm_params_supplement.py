"""Run only the five-seed additions for the GGPL-GTM parameter experiment.

The original 399-task experiment remains read-only under ``--legacy-root``.
This runner writes every new attempt below ``--output-root`` so a retry cannot
overwrite an earlier log, checkpoint, prediction, or completed result.
"""

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


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
LEGACY_SEEDS = (3, 4)
EXTRA_SEEDS = (0, 1, 2, 3, 4)
EXTRAS = {
    "extra_tau8": {"graph_temperature": 8.0},
    "extra_tau16": {"graph_temperature": 16.0},
    "extra_r4_tau2": {"graph_dynamic_rank": 4, "graph_temperature": 2.0},
    "extra_r4_tau4": {"graph_dynamic_rank": 4, "graph_temperature": 4.0},
    "extra_r64_tau2": {"graph_dynamic_rank": 64, "graph_temperature": 2.0},
    "extra_r64_tau4": {"graph_dynamic_rank": 64, "graph_temperature": 4.0},
}
EXPECTED_LEGACY_OVERRIDES = {
    "baseline": {},
    "K_2": {"num_breakpoints": 2}, "K_4": {"num_breakpoints": 4},
    "K_16": {"num_breakpoints": 16}, "K_32": {"num_breakpoints": 32},
    "r_4": {"graph_dynamic_rank": 4}, "r_8": {"graph_dynamic_rank": 8},
    "r_32": {"graph_dynamic_rank": 32}, "r_64": {"graph_dynamic_rank": 64},
    "tau_0.25": {"graph_temperature": 0.25}, "tau_0.5": {"graph_temperature": 0.5},
    "tau_2": {"graph_temperature": 2.0}, "tau_4": {"graph_temperature": 4.0},
    "d_128": {"d_token": 128}, "d_256": {"d_token": 256}, "d_512": {"d_token": 512},
    "L_2": {"n_layers": 2}, "L_3": {"n_layers": 3}, "L_4": {"n_layers": 4},
}
REQUIRED_PARAMETERS = (
    "num_breakpoints", "graph_dynamic_rank", "graph_temperature", "d_token", "n_layers",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--group", choices=("all", "legacy-seeds", "extras"), default="all")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--configs", nargs="+", default=None,
                   help="Optional config IDs within the selected group.")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--legacy-root", default="results/ggpl_gtm_parameter")
    p.add_argument("--output-root", default="results/ggpl_gtm_parameter_supplement")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    return p


def validate_parameters(parameters, context):
    missing = set(REQUIRED_PARAMETERS) - set(parameters)
    if missing:
        raise SystemExit(f"{context} is missing model parameters: {sorted(missing)}")


def source_fingerprint(repo):
    files = (repo / "models" / "ggpl_gtm.py", repo / "models" / "ggpl_tmlp.py", repo / "main.py")
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def load_legacy(legacy_root):
    manifest = read_json(legacy_root / "manifest.json")
    if not manifest or not isinstance(manifest.get("tasks"), list):
        raise SystemExit(f"Missing or invalid legacy manifest: {legacy_root / 'manifest.json'}")
    tasks = manifest["tasks"]
    if len(tasks) != 399:
        raise SystemExit(f"Expected 399 legacy tasks, found {len(tasks)}")
    grouped = {}
    for task in tasks:
        if task.get("model") != "ggpl_gtm":
            raise SystemExit("Legacy manifest does not use the formal ggpl_gtm model")
        key = task.get("config_id")
        grouped.setdefault(key, []).append(task)
    if len(grouped) != 19 or set(grouped) != {
        "baseline", "K_2", "K_4", "K_16", "K_32", "r_4", "r_8", "r_32", "r_64",
        "tau_0.25", "tau_0.5", "tau_2", "tau_4", "d_128", "d_256", "d_512",
        "L_2", "L_3", "L_4",
    }:
        raise SystemExit("Legacy manifest does not contain the expected 19 unique configurations")
    configs = {}
    for config_id, entries in grouped.items():
        parameters = entries[0].get("parameters", {})
        validate_parameters(parameters, f"legacy config {config_id}")
        if any(entry.get("parameters") != parameters for entry in entries):
            raise SystemExit(f"Legacy config {config_id} has inconsistent saved parameters")
        configs[config_id] = {"parameters": copy.deepcopy(parameters), "template": entries[0]}
    baseline = configs["baseline"]["parameters"]
    expected = {
        "num_breakpoints": 8, "graph_dynamic_rank": 16,
        "graph_temperature": 1.0, "d_token": 1024, "n_layers": 1,
    }
    if baseline != expected:
        raise SystemExit(f"Legacy baseline differs from the documented design: {baseline}")
    for config_id, override in EXPECTED_LEGACY_OVERRIDES.items():
        expected_parameters = {**baseline, **override}
        if configs[config_id]["parameters"] != expected_parameters:
            raise SystemExit(
                f"Legacy config {config_id} differs from the documented one-factor design: "
                f"{configs[config_id]['parameters']}"
            )
    return manifest, configs


def valid_legacy(task):
    output = Path(task["output_dir"])
    marker = read_json(output / "parameter_result.json")
    metadata = read_json(output / "parameter_task.json")
    prediction = read_json(output / "prediction.json")
    if not marker or not metadata or not prediction:
        return False, "missing result, task metadata, or prediction"
    if metadata.get("config_fingerprint") != task.get("config_fingerprint"):
        return False, "legacy task fingerprint mismatch"
    if marker.get("config_fingerprint") != task.get("config_fingerprint"):
        return False, "legacy result fingerprint mismatch"
    if not isinstance((prediction.get("metrics") or {}).get("rmse"), (int, float)):
        return False, "legacy test RMSE is invalid"
    return True, None


def build_supplement_manifest(repo, legacy_root, configs, legacy_manifest):
    common = {
        "model": "ggpl_gtm",
        "legacy_root": str(legacy_root),
        "legacy_manifest_created_at": legacy_manifest.get("created_at"),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True
        ).stdout.strip() or None,
        "source_fingerprint": source_fingerprint(repo),
    }
    tasks = []
    for config_id, source in configs.items():
        template = source["template"]
        for seed in LEGACY_SEEDS:
            task = {
                **common, "group": "legacy-seeds", "config_id": config_id,
                "dataset": None, "training_seed": seed,
                "parameters": copy.deepcopy(source["parameters"]),
                "data_seed": template["data_seed"], "lr": template["lr"],
                "batch_size": template["batch_size"],
                "base_model_config": copy.deepcopy(template["base_model_config"]),
                "base_training_config": copy.deepcopy(template["base_training_config"]),
            }
            tasks.append(task)
    baseline_template = configs["baseline"]["template"]
    baseline = configs["baseline"]["parameters"]
    for config_id, changes in EXTRAS.items():
        parameters = {**baseline, **changes}
        for seed in EXTRA_SEEDS:
            task = {
                **common, "group": "extras", "config_id": config_id,
                "dataset": None, "training_seed": seed, "parameters": parameters,
                "data_seed": baseline_template["data_seed"], "lr": baseline_template["lr"],
                "batch_size": baseline_template["batch_size"],
                "base_model_config": copy.deepcopy(baseline_template["base_model_config"]),
                "base_training_config": copy.deepcopy(baseline_template["base_training_config"]),
            }
            tasks.append(task)
    expanded = []
    for prototype in tasks:
        for dataset in DATASETS:
            task = copy.deepcopy(prototype)
            task["dataset"] = dataset
            identity = {key: value for key, value in task.items() if key not in {"status", "output_dir", "config_fingerprint"}}
            task["config_fingerprint"] = fingerprint(identity)
            task["status"] = "pending"
            expanded.append(task)
    if len(expanded) != 476:
        raise AssertionError(f"Expected 476 supplement tasks, got {len(expanded)}")
    return {
        "schema_version": 1, "experiment": "ggpl_gtm_parameter_supplement",
        "created_at": time.time(), "legacy_root": str(legacy_root),
        "expected_legacy_tasks": 399, "expected_supplement_tasks": 476,
        "expected_all_tasks": 875, "tasks": expanded,
    }


def task_root(root, task):
    return root / task["group"] / task["config_id"] / task["dataset"] / f"seed_{task['training_seed']}"


def valid_supplement(task, root):
    root_dir = task_root(root, task)
    marker = read_json(root_dir / "parameter_result.json")
    if not marker:
        return False, "no completed result"
    if marker.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "completed-result fingerprint conflict"
    attempt = marker.get("attempt_dir")
    if not isinstance(attempt, str):
        return False, "completed result has no attempt directory"
    attempt_dir = Path(attempt)
    metadata = read_json(attempt_dir / "parameter_task.json")
    prediction = read_json(attempt_dir / "prediction.json")
    if not metadata or metadata.get("config_fingerprint") != task["config_fingerprint"]:
        return False, "attempt task fingerprint conflict"
    if not prediction or not isinstance((prediction.get("metrics") or {}).get("rmse"), (int, float)):
        return False, "attempt test RMSE is invalid"
    return True, None


def next_attempt(root_dir):
    attempts = root_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices = [int(path.name.split("_")[-1]) for path in attempts.glob("attempt_*") if path.name.split("_")[-1].isdigit()]
    return attempts / f"attempt_{max(indices, default=0) + 1:03d}"


def acquire_lock(root_dir):
    root_dir.mkdir(parents=True, exist_ok=True)
    lock = root_dir / ".task.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(handle, str(os.getpid()).encode())
    return lock, handle


def release_lock(lock_handle):
    if lock_handle is None:
        return
    lock, handle = lock_handle
    os.close(handle)
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def materialize_attempt(task, root):
    root_dir = task_root(root, task)
    attempt_dir = next_attempt(root_dir)
    attempt_dir.mkdir(parents=True)
    config = {"model": copy.deepcopy(task["base_model_config"]), "training": copy.deepcopy(task["base_training_config"])}
    config["model"].update(task["parameters"])
    config["model"].update({
        "model_name": "ggpl_gtm",
        "breakpoint_cache_dir": str(attempt_dir / "breakpoint_cache"),
    })
    config["training"].update({"lr": task["lr"], "batch_size": task["batch_size"]})
    if any(config["model"].get(key) != value for key, value in task["parameters"].items()):
        raise RuntimeError(f"Materialized config does not preserve parameters for {task['config_id']}")
    config_path = attempt_dir / "parameter_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(attempt_dir / "parameter_task.json", task)
    return attempt_dir, config_path


def command(task, config_path, attempt_dir, gpu):
    parameters = task["parameters"]
    return [
        sys.executable, "-u", "main.py", "--model", "ggpl_gtm", "--dataset", task["dataset"],
        "--device", "cuda", "--gpu", str(gpu), "--seed", str(task["training_seed"]),
        "--data_seed", str(task["data_seed"]), "--batch_size", str(task["batch_size"]),
        "--lr", str(task["lr"]), "--d_token", str(parameters["d_token"]),
        "--n_layers", str(parameters["n_layers"]), "--config", str(config_path),
        "--output_dir", str(attempt_dir),
    ]


def main():
    args = parser().parse_args()
    if set(args.datasets) - set(DATASETS):
        raise SystemExit(f"Unknown datasets: {sorted(set(args.datasets) - set(DATASETS))}")
    repo = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root).resolve()
    root = Path(args.output_root).resolve()
    legacy_manifest, configs = load_legacy(legacy_root)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest is None:
        manifest = build_supplement_manifest(repo, legacy_root, configs, legacy_manifest)
        atomic_json(manifest_path, manifest)
    if len(manifest.get("tasks", [])) != 476 or manifest.get("legacy_root") != str(legacy_root):
        raise SystemExit("Supplement manifest conflicts with the requested experiment roots")

    selected_groups = {"legacy-seeds", "extras"} if args.group == "all" else {args.group}
    selected = [task for task in manifest["tasks"] if task["group"] in selected_groups and task["dataset"] in args.datasets]
    if args.configs:
        selected = [task for task in selected if task["config_id"] in args.configs]
    unknown_configs = set(args.configs or ()) - {task["config_id"] for task in manifest["tasks"]}
    if unknown_configs:
        raise SystemExit(f"Unknown configs: {sorted(unknown_configs)}")

    legacy_invalid = []
    for task in legacy_manifest["tasks"]:
        valid, reason = valid_legacy(task)
        if not valid:
            legacy_invalid.append((task["config_id"], task["dataset"], task["training_seed"], reason))
    counts = {"completed": 0, "pending": 0, "failed": 0, "running": 0, "needs_review": 0}
    runnable = []
    for task in selected:
        valid, reason = valid_supplement(task, root)
        root_dir = task_root(root, task)
        if valid:
            task["status"] = "completed"
            counts["completed"] += 1
        elif (root_dir / ".task.lock").exists():
            task["status"] = "running"
            counts["running"] += 1
        elif (root_dir / "parameter_result.json").exists():
            task["status"] = "needs_review"
            task["failure_reason"] = reason
            counts["needs_review"] += 1
        elif task.get("status") == "failed" and not args.retry_failed:
            counts["failed"] += 1
        elif task.get("status") == "needs_review":
            counts["needs_review"] += 1
        else:
            task["status"] = "pending"
            counts["pending"] += 1
            runnable.append(task)
    atomic_json(manifest_path, manifest)
    print(f"supplement design: 19 legacy configs x seeds 3,4 = 266; 6 extras x seeds 0..4 = 210; total=476")
    print(f"selected={len(selected)} completed={counts['completed']} pending={counts['pending']} failed={counts['failed']} running={counts['running']} needs_review={counts['needs_review']}")
    print(f"legacy audit: valid={399 - len(legacy_invalid)} invalid_or_missing={len(legacy_invalid)} (not added to this queue)")
    for item in legacy_invalid:
        print("legacy-needs-review:", *item)
    for task in runnable:
        print("pending:", task["group"], task["config_id"], task["dataset"], f"seed={task['training_seed']}", task_root(root, task))
    if args.dry_run:
        return

    children, stopping = {}, False

    def stop_children(*_):
        nonlocal stopping
        stopping = True
        for process, _, _ in children.values():
            process.terminate()

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    while (runnable and not stopping) or children:
        while runnable and len(children) < len(args.gpus) and not stopping:
            task = runnable.pop(0)
            root_dir = task_root(root, task)
            lock_handle = acquire_lock(root_dir)
            if lock_handle is None:
                task["status"] = "needs_review"
                task["failure_reason"] = "task lock already exists"
                atomic_json(manifest_path, manifest)
                print(f"needs_review: locked {task['config_id']} {task['dataset']} seed={task['training_seed']}")
                continue
            gpu = args.gpus[len(children) % len(args.gpus)]
            attempt_dir, config_path = materialize_attempt(task, root)
            task.update(status="running", started_at=time.time(), gpu=gpu, attempt_dir=str(attempt_dir))
            atomic_json(manifest_path, manifest)
            log = (attempt_dir / "training.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command(task, config_path, attempt_dir, gpu), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, (task, log), lock_handle)
        for pid, (process, (task, log), lock_handle) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            children.pop(pid)
            release_lock(lock_handle)
            task["finished_at"], task["return_code"] = time.time(), code
            if code == 0:
                prediction = read_json(Path(task["attempt_dir"]) / "prediction.json")
                rmse = (prediction or {}).get("metrics", {}).get("rmse")
                if isinstance(rmse, (int, float)):
                    atomic_json(task_root(root, task) / "parameter_result.json", {
                        "config_fingerprint": task["config_fingerprint"], "attempt_dir": task["attempt_dir"],
                        "test_rmse": rmse, "completed_at": time.time(),
                    })
                    task["status"], task["failure_reason"] = "completed", None
                else:
                    task["status"], task["failure_reason"] = "failed", "prediction.json has no valid test RMSE"
            else:
                task["status"], task["failure_reason"] = "failed", f"main.py exited with {code}"
            atomic_json(manifest_path, manifest)
            print(f"{task['status']}: {task['group']} {task['config_id']} {task['dataset']} seed={task['training_seed']}", flush=True)
        if children:
            time.sleep(0.5)
    if stopping:
        atomic_json(manifest_path, manifest)
        raise SystemExit("Interrupted: only child processes started by this scheduler were terminated")


if __name__ == "__main__":
    main()
