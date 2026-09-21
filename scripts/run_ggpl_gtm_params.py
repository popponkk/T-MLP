"""Launch resumable, one-factor GGPL-GTM parameter-sensitivity experiments."""

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


DATASETS = [
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
]
SEEDS = [0, 1, 2]
BASELINE = {
    "num_breakpoints": 8, "graph_dynamic_rank": 16,
    "graph_temperature": 1.0, "d_token": 1024, "n_layers": 1,
}
PARAMETER_CONFIGS = {
    "baseline": {},
    "K_2": {"num_breakpoints": 2}, "K_4": {"num_breakpoints": 4},
    "K_16": {"num_breakpoints": 16}, "K_32": {"num_breakpoints": 32},
    "r_4": {"graph_dynamic_rank": 4}, "r_8": {"graph_dynamic_rank": 8},
    "r_32": {"graph_dynamic_rank": 32}, "r_64": {"graph_dynamic_rank": 64},
    "tau_0.25": {"graph_temperature": 0.25},
    "tau_0.5": {"graph_temperature": 0.5},
    "tau_2": {"graph_temperature": 2.0}, "tau_4": {"graph_temperature": 4.0},
    "d_128": {"d_token": 128}, "d_256": {"d_token": 256},
    "d_512": {"d_token": 512},
    "L_2": {"n_layers": 2}, "L_3": {"n_layers": 3}, "L_4": {"n_layers": 4},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint(repo):
    files = [repo / "models" / "ggpl_gtm.py", repo / "models" / "ggpl_tmlp.py", repo / "main.py"]
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def git_revision(repo):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_design():
    if len(PARAMETER_CONFIGS) != 19:
        raise AssertionError("Expected exactly 19 unique parameter configurations")
    for name, override in PARAMETER_CONFIGS.items():
        if name == "baseline":
            if override:
                raise AssertionError("baseline must not override a parameter")
        elif len(override) != 1:
            raise AssertionError(f"{name} must change exactly one parameter")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--configs", nargs="+", default=list(PARAMETER_CONFIGS))
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--output-root", default="results/ggpl_gtm_parameter")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    return p


def build_manifest(args, repo, root, default_config):
    model_config = default_config["model"]
    training_config = default_config.get("training", {})
    common = {
        "model": "ggpl_gtm", "data_seed": args.data_seed,
        "lr": args.lr, "batch_size": args.batch_size,
        "base_model_config": model_config, "base_training_config": training_config,
        "git_revision": git_revision(repo), "source_fingerprint": source_fingerprint(repo),
    }
    tasks = []
    for config_id, override in PARAMETER_CONFIGS.items():
        parameters = {**BASELINE, **override}
        for dataset in DATASETS:
            for seed in SEEDS:
                output_dir = root / config_id / dataset / f"seed_{seed}"
                condition = {**common, "config_id": config_id, "dataset": dataset,
                             "training_seed": seed, "parameters": parameters}
                tasks.append({**condition, "output_dir": str(output_dir),
                              "config_fingerprint": sha256(condition), "status": "pending"})
    return {"schema_version": 1, "created_at": time.time(), "experiment": "ggpl_gtm_parameter",
            "baseline": BASELINE, "tasks": tasks}


def valid_prediction(task):
    prediction = Path(task["output_dir"]) / "prediction.json"
    metadata = Path(task["output_dir"]) / "parameter_task.json"
    completed = Path(task["output_dir"]) / "parameter_result.json"
    if not prediction.exists() or not metadata.exists() or not completed.exists():
        return False, "prediction.json, parameter_task.json, or parameter_result.json is missing"
    try:
        payload = json.loads(prediction.read_text(encoding="utf-8"))
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        rmse = (payload.get("metrics") or {}).get("rmse")
        if not isinstance(rmse, (int, float)):
            return False, "test RMSE is missing or invalid"
        if meta.get("config_fingerprint") != task["config_fingerprint"]:
            return False, "configuration fingerprint conflict"
        if read_completed_result(completed).get("config_fingerprint") != task["config_fingerprint"]:
            return False, "completed-result fingerprint conflict"
    except (OSError, ValueError, TypeError) as error:
        return False, f"invalid result metadata: {error}"
    return True, None


def read_completed_result(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_result(task):
    output_dir = Path(task["output_dir"])
    prediction = json.loads((output_dir / "prediction.json").read_text(encoding="utf-8"))
    rmse = (prediction.get("metrics") or {}).get("rmse")
    if not isinstance(rmse, (int, float)):
        return False, "test RMSE is missing or invalid"
    atomic_json(output_dir / "parameter_result.json", {
        "config_fingerprint": task["config_fingerprint"],
        "test_rmse": rmse,
        "prediction_file": "prediction.json",
        "completed_at": time.time(),
    })
    return True, None


def materialize_task(task, default_config):
    output_dir = Path(task["output_dir"])
    config = copy.deepcopy(default_config)
    config.setdefault("model", {}).update(task["parameters"])
    config["model"].update({
        "model_name": "ggpl_gtm",
        # Each task owns its initialization cache; no incompatible reuse is possible.
        "breakpoint_cache_dir": str(output_dir / "breakpoint_cache"),
    })
    config.setdefault("training", {}).update({"lr": task["lr"], "batch_size": task["batch_size"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "parameter_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(output_dir / "parameter_task.json", task)
    return config_path


def task_command(task, config_path, repo, gpu):
    parameters = task["parameters"]
    return [
        sys.executable, "main.py", "--model", "ggpl_gtm", "--dataset", task["dataset"],
        "--device", "cuda", "--gpu", str(gpu), "--seed", str(task["training_seed"]),
        "--data_seed", str(task["data_seed"]), "--batch_size", str(task["batch_size"]),
        "--lr", str(task["lr"]), "--d_token", str(parameters["d_token"]),
        "--n_layers", str(parameters["n_layers"]), "--config", str(config_path),
        "--output_dir", task["output_dir"],
    ]


def main():
    args = parser().parse_args()
    validate_design()
    unknown = set(args.datasets) - set(DATASETS)
    unknown_configs = set(args.configs) - set(PARAMETER_CONFIGS)
    if unknown or unknown_configs:
        raise SystemExit(f"Unknown datasets={sorted(unknown)} configs={sorted(unknown_configs)}")
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).resolve()
    config_path = repo / "configs" / "default" / "ggpl_gtm.yaml"
    default_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("baseline") != BASELINE or len(manifest.get("tasks", [])) != 399:
            raise SystemExit("Existing manifest conflicts with the fixed 399-task parameter plan")
        for task in manifest["tasks"]:
            if task["status"] == "running":
                task["status"] = "interrupted"
                task["failure_reason"] = "previous scheduler exited before task completion"
    else:
        manifest = build_manifest(args, repo, root, default_config)
    atomic_json(manifest_path, manifest)
    selected = [task for task in manifest["tasks"] if task["dataset"] in args.datasets
                and task["training_seed"] in args.seeds and task["config_id"] in args.configs]
    print(f"plan: 19 configurations, {len(manifest['tasks'])} total tasks, {len(selected)} selected")
    for task in selected:
        print(task["config_id"], task["dataset"], task["training_seed"], task["parameters"], task["output_dir"])
    if args.dry_run:
        return

    pending = []
    for task in selected:
        complete, reason = valid_prediction(task)
        if task["status"] == "completed":
            if complete:
                if args.resume:
                    continue
                raise SystemExit(f"Completed task exists at {task['output_dir']}; use --resume to skip it")
            task["status"] = "failed"
            task["failure_reason"] = reason
            if not args.retry_failed:
                continue
        if task["status"] in {"failed", "interrupted"} and not args.retry_failed:
            continue
        pending.append(task)
    atomic_json(manifest_path, manifest)

    children, stopping = {}, False
    def stop_children(*_):
        nonlocal stopping
        stopping = True
        for process, _ in children.values():
            process.terminate()
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    # Once interrupted, pending tasks must not keep the scheduler alive.
    while (pending and not stopping) or children:
        while pending and len(children) < len(args.gpus) and not stopping:
            task = pending.pop(0)
            gpu = args.gpus[len(children) % len(args.gpus)]
            config_file = materialize_task(task, default_config)
            command = task_command(task, config_file, repo, gpu)
            log_file = Path(task["output_dir"]) / "training.log"
            log = log_file.open("w", encoding="utf-8")
            task.update(status="running", started_at=time.time(), gpu=gpu, command=command)
            atomic_json(manifest_path, manifest)
            process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children[process.pid] = (process, (task, log))
        for pid, (process, (task, log)) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            children.pop(pid)
            finalized, final_reason = (
                finalize_result(task) if code == 0 else (False, f"main.py exited with {code}")
            )
            complete, reason = valid_prediction(task) if finalized else (False, final_reason)
            task["finished_at"] = time.time()
            task["return_code"] = code
            task["status"] = "completed" if code == 0 and complete else "failed"
            task["failure_reason"] = None if task["status"] == "completed" else reason
            atomic_json(manifest_path, manifest)
            print(f"{task['status']}: {task['config_id']} {task['dataset']} seed={task['training_seed']}")
        if children:
            time.sleep(0.5)
    if stopping:
        atomic_json(manifest_path, manifest)
        raise SystemExit("Interrupted: only child processes started by this scheduler were terminated")


if __name__ == "__main__":
    main()
