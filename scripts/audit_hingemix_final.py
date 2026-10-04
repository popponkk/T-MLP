"""Read-only audit of historical baseline results for the formal HingeMix study."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASETS = ("hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161", "raiderstream", "stream", "cachesweep")
BASELINES = {"LightGBM": "lightgbm", "XGBoost": "xgboost", "CatBoost": "catboost", "ExcelFormer": "excel-former", "FT-Transformer": "ft-transformer", "DCNv2": "dcnv2", "NODE": "node", "MLP": "mlp", "AutoInt": "autoint", "TabM": "tabm"}
STATUS_UNKNOWN = "\u5f85\u6838\u67e5\uff0c\u8bc1\u636e\u4e0d\u8db3"
STATUS_SEED_UNKNOWN = "\u627e\u5230\u7ed3\u679c\uff0c\u4f46\u79cd\u5b50\u672a\u77e5"
STATUS_INVALID = "\u5df2\u786e\u8ba4\u6761\u4ef6\u4e0d\u4e00\u81f4\u6216\u8fd0\u884c\u65e0\u6548"


def read_json(path, issues):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(issue("json_read_error", path, str(exc)))
        return None


def issue(kind, path, detail, model="", dataset=""):
    return {"issue_type": kind, "path": str(path), "model": model, "dataset": dataset, "detail": detail}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exact_component(parts, name):
    # Accept only a model directory or its explicit seed suffix, never a substring.
    return any(part == name or re.fullmatch(re.escape(name) + r"[_-]seed[_-]?\d+", part) for part in parts)


def path_identity(run_dir, root):
    return str(run_dir.resolve()).replace("\\", "/")


def discover_run_dirs(root, issues):
    if not root.is_dir():
        return [], 0
    run_dirs, seen_dirs, scanned = [], set(), 0
    for current, dirs, files in os.walk(root, followlinks=False):
        scanned += len(files)
        if "prediction.json" not in files and "results.json" not in files:
            continue
        directory = Path(current)
        resolved = path_identity(directory, root)
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        run_dirs.append(directory)
    return run_dirs, scanned


def locate_component(run_dir, root, values):
    try:
        parts = [part.lower() for part in run_dir.relative_to(root).parts]
    except ValueError:
        parts = [part.lower() for part in run_dir.parts]
    matches = [value for value in values if value in parts]
    return matches[0] if len(matches) == 1 else None


def nearby_metadata(run_dir, root, issues):
    names = ("parameter_task.json", "task.json", "parameter_result.json", "manifest.json", "config.json", "final_config.yaml", "parameter_config.yaml", "config.yaml")
    found = []
    current = run_dir
    while True:
        for name in names:
            candidate = current / name
            if candidate.is_file():
                found.append(candidate)
        if current == root or current.parent == current:
            break
        current = current.parent
    return found


def seed_values(value, key=""):
    values = []
    if isinstance(value, dict):
        for name, child in value.items():
            lowered = name.lower()
            if lowered in {"training_seed", "train_seed", "seed"} and isinstance(child, int):
                values.append((lowered, child))
            values.extend(seed_values(child, lowered))
    elif isinstance(value, list):
        for child in value:
            values.extend(seed_values(child, key))
    return values


def named_values(value, wanted):
    found = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name.lower() in wanted and isinstance(child, str):
                found.append(child)
            found.extend(named_values(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(named_values(child, wanted))
    return found


def extract_evidence(run_dir, root, issues):
    seeds, configs, metadata_paths, model_names = [], [], [], []
    # Explicit path seed is evidence, but not silently promoted to a training seed.
    for part in run_dir.relative_to(root).parts:
        match = re.fullmatch(r"seed[_-]?(\d+)", part.lower())
        if match:
            seeds.append(("path_seed", int(match.group(1))))
    for path in nearby_metadata(run_dir, root, issues):
        metadata_paths.append(str(path))
        if path.suffix == ".json":
            data = read_json(path, issues)
            if data is not None:
                seeds.extend(seed_values(data))
                model_names.extend(named_values(data, {"model_name", "model"}))
        else:
            # A config is retained as provenance; YAML parsing is intentionally
            # not guessed when the optional dependency is unavailable.
            configs.append(str(path))
    values = {seed for _, seed in seeds}
    seed = next(iter(values)) if len(values) == 1 else None
    seed_status = "known" if seed is not None else ("conflict" if len(values) > 1 else "unknown")
    return seed, seed_status, sorted(set(metadata_paths)), sorted(set(configs)), seeds, sorted(set(model_names))


def test_metric(prediction, issues, path, model, dataset):
    if not isinstance(prediction, dict):
        return None, None
    name, metric = prediction.get("metric_name"), prediction.get("metric")
    nested = (prediction.get("metrics") or {}).get("rmse") if isinstance(prediction.get("metrics"), dict) else None
    candidates = []
    if name == "rmse" and finite(metric):
        candidates.append((metric, "prediction.metric (metric_name=rmse)"))
    if finite(nested):
        candidates.append((nested, "prediction.metrics.rmse"))
    unique = {value for value, _ in candidates}
    if len(unique) > 1:
        issues.append(issue("test_metric_conflict", path, "prediction.metric and prediction.metrics.rmse disagree", model, dataset))
        return None, None
    return candidates[0] if candidates else (None, None)


def validation_metric(history):
    if not isinstance(history, dict):
        return None, None
    val = history.get("val")
    if not isinstance(val, dict) or val.get("metric_name") != "rmse":
        return None, None
    value = val.get("best_metric")
    return (value, "results.val.best_metric") if finite(value) else (None, None)


def discover_records(root):
    issues, records = [], []
    run_dirs, scanned = discover_run_dirs(root, issues)
    for run_dir in run_dirs:
        model = locate_component(run_dir, root, set(BASELINES.values()))
        dataset = locate_component(run_dir, root, set(DATASETS))
        if model is None or dataset is None:
            issues.append(issue("unrecognized_model_or_dataset", run_dir, "exact baseline model and dataset path components are required", model or "", dataset or ""))
            continue
        prediction_path, history_path = run_dir / "prediction.json", run_dir / "results.json"
        prediction = read_json(prediction_path, issues) if prediction_path.is_file() else None
        history = read_json(history_path, issues) if history_path.is_file() else None
        test_rmse, test_source = test_metric(prediction, issues, prediction_path, model, dataset)
        val_rmse, val_source = validation_metric(history)
        seed, seed_status, metadata, configs, seed_evidence, metadata_models = extract_evidence(run_dir, root, issues)
        if seed_status == "conflict":
            issues.append(issue("seed_conflict", run_dir, repr(seed_evidence), model, dataset))
        if metadata_models and model not in metadata_models:
            issues.append(issue("model_metadata_path_conflict", run_dir, repr(metadata_models), model, dataset))
        records.append({
            "model_registration": model, "dataset": dataset, "run_dir": str(run_dir.resolve()),
            "prediction_path": str(prediction_path) if prediction_path.is_file() else "",
            "results_path": str(history_path) if history_path.is_file() else "",
            "test_rmse": test_rmse, "test_rmse_source": test_source or "",
            "validation_rmse": val_rmse, "validation_rmse_source": val_source or "",
            "training_seed": "" if seed is None else seed, "seed_status": seed_status,
            "seed_evidence": repr(seed_evidence), "metadata_paths": ";".join(metadata),
            "config_paths": ";".join(configs), "metadata_models": ";".join(metadata_models),
            "data_split_evidence": "not found", "target_scale_evidence": "not found",
        })
    return records, issues, scanned, len(run_dirs)


def baseline_audit(repo, records, issues):
    rows, pending = [], []
    by_key = defaultdict(list)
    for record in records:
        by_key[(record["model_registration"], record["dataset"])].append(record)
    for display, registration in BASELINES.items():
        for dataset in DATASETS:
            candidates = by_key[(registration, dataset)]
            valid = [r for r in candidates if finite(r["test_rmse"])]
            known = [int(r["training_seed"]) for r in valid if r["seed_status"] == "known"]
            unknown = [r for r in valid if r["seed_status"] != "known"]
            configurations = {r["config_paths"] for r in valid if r["config_paths"]}
            if not candidates:
                status, reason = STATUS_UNKNOWN, "No matching historical run was located; absence is not proof that it was never run."
            elif not valid:
                status, reason = STATUS_INVALID, "Located run directories but no unambiguous test RMSE in prediction.json."
            elif unknown:
                status, reason = STATUS_SEED_UNKNOWN, "Valid test result exists, but training seed is absent or conflicting."
            elif len(set(known)) < 5:
                status, reason = STATUS_UNKNOWN, "Known seeds are incomplete, but no manifest proves that missing seeds require reruns."
            elif len(configurations) > 1:
                status, reason = STATUS_UNKNOWN, "Multiple configuration sources exist for the same model/dataset; do not merge them."
            else:
                status, reason = STATUS_UNKNOWN, "Five seeds found, but data version, split identity, target scale, and historical configuration comparability remain unverified."
            rows.append({
                "model": display, "registration": registration, "dataset": dataset,
                "result_paths": ";".join(r["run_dir"] for r in candidates),
                "valid_seeds": ";".join(map(str, sorted(set(known)))),
                "unknown_seed_records": str(len(unknown)),
                "data_version_check": "unable_to_confirm", "split_check": "unable_to_confirm",
                "rmse_scale_check": "unable_to_confirm", "config_sources": ";".join(sorted(configurations)),
                "status": status, "reason": reason,
                "next_step": "Locate manifest/config/log and verify split plus target-scale evidence before reuse or rerun decision.",
            })
    return rows, pending


def ablation_rows():
    return [
        {"ablation": "full", "display_name": "HingeMix", "tokenizer": "GGPLTokenizer", "graph": True, "channel": True, "readout": "CLS x[:, 0]", "category": "A", "static_evidence": "Graph attention @ H mixes tokens; source: models/hingemix.py:53-74,133-140", "runtime_validation": "not run by this audit"},
        {"ablation": "no_graph", "display_name": "HingeMix-no_graph", "tokenizer": "GGPLTokenizer", "graph": False, "channel": True, "readout": "CLS x[:, 0]", "category": "B", "static_evidence": "CLS is learned constant; channel is token-wise; source: models/ggpl_tmlp.py:119-150; models/hingemix_ablation.py:116-137,211-218", "runtime_validation": "not run by this audit"},
        {"ablation": "linear", "display_name": "HingeMix-linear", "tokenizer": "shared nn.Linear(1,d_token)", "graph": True, "channel": True, "readout": "CLS x[:, 0]", "category": "A", "static_evidence": "Graph attention @ H mixes tokens; source: models/hingemix_ablation.py:39-64,116-137,211-218", "runtime_validation": "not run by this audit"},
        {"ablation": "linear_no_graph", "display_name": "HingeMix-linear_no_graph", "tokenizer": "shared nn.Linear(1,d_token)", "graph": False, "channel": True, "readout": "CLS x[:, 0]", "category": "B", "static_evidence": "CLS is learned constant; channel is token-wise; source: models/hingemix_ablation.py:39-64,116-137,211-218", "runtime_validation": "not run by this audit"},
    ]


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "results"
        # Prediction and history merge into one run; unknown seed must survive.
        run = root / "mlp" / "hpcg2"
        run.mkdir(parents=True)
        (run / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 1.5, "metrics": {"rmse": 1.5}}))
        (run / "results.json").write_text(json.dumps({"val": {"metric_name": "rmse", "best_metric": 1.2}}))
        seeded = root / "dcnv2" / "seed_3" / "hpgmg3"
        seeded.mkdir(parents=True)
        (seeded / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 2.0}))
        false_match = root / "ggpl_tmlp" / "hpcg2"
        false_match.mkdir(parents=True)
        (false_match / "prediction.json").write_text(json.dumps({"metric_name": "rmse", "metric": 9.0}))
        records, issues, _, _ = discover_records(root)
        assert len(records) == 2, records
        assert any(r["model_registration"] == "mlp" and r["seed_status"] == "unknown" for r in records)
        assert any(r["model_registration"] == "dcnv2" and r["training_seed"] == 3 for r in records)
        assert all(r["model_registration"] != "mlp" or "ggpl_tmlp" not in r["run_dir"] for r in records)
        mlp_record = next(r for r in records if r["model_registration"] == "mlp")
        assert mlp_record["test_rmse_source"].startswith("prediction")
        assert mlp_record["validation_rmse_source"].startswith("results")
    return "synthetic checks passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="audit/hingemix_final_audit_v2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(self_test())
        return
    repo, root, output = Path(__file__).resolve().parents[1], Path(args.results_root).resolve(), Path(args.output)
    records, issues, scanned, run_dirs = discover_records(root)
    rows, pending = baseline_audit(repo, records, issues)
    write_csv(output / "discovered_runs.csv", records, list(records[0]) if records else ["model_registration", "dataset", "run_dir", "prediction_path", "results_path", "test_rmse", "test_rmse_source", "validation_rmse", "validation_rmse_source", "training_seed", "seed_status", "seed_evidence", "metadata_paths", "config_paths", "metadata_models", "data_split_evidence", "target_scale_evidence"])
    write_csv(output / "baseline_audit.csv", rows, list(rows[0]))
    write_csv(output / "pending_runs.csv", pending, ["model", "dataset", "reason", "action"])
    write_csv(output / "scan_issues.csv", issues, ["issue_type", "path", "model", "dataset", "detail"])
    write_csv(output / "ablation_checks.csv", ablation_rows(), list(ablation_rows()[0]))
    report = f"# HingeMix audit v2\n\nGenerated: {datetime.now(timezone.utc).isoformat()}\n\nResults root: `{root}`\n\nExists: `{root.is_dir()}`\n\nScanned files: {scanned}\n\nRun directories with prediction/results JSON: {run_dirs}\n\nMatched baseline run records: {len(records)}\n\nParse/identity issues: {len(issues)}\n\nThis report is generated from the scan above. Static ablation conclusions are recorded separately in `ablation_checks.csv`; no forward validation is claimed by this script.\n"
    (output / "audit_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote audit files to {output}; matched={len(records)} issues={len(issues)}")


if __name__ == "__main__":
    main()
