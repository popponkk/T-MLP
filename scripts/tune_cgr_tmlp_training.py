import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_DATASETS = [
    "Ailerons",
    "diamonds",
    "medical_charges",
    "california",
    "house_sales",
]


def tag_float(value):
    return str(value).replace("-", "m").replace(".", "p")


def build_grid(mode):
    spec_output_scales = [0.09, 0.10, 0.11]
    gamma_init_biases = [-2.4, -2.6, -2.8]
    topk_ratios = [0.15] if mode == "default9" else [0.10, 0.15, 0.20]
    return [
        {
            "spec_output_scale": spec_output_scale,
            "gamma_init_bias": gamma_init_bias,
            "topk_ratio": topk_ratio,
        }
        for spec_output_scale, gamma_init_bias, topk_ratio in itertools.product(
            spec_output_scales,
            gamma_init_biases,
            topk_ratios,
        )
    ]


def write_config(base_config, params, output_file):
    config = {
        "model": dict(base_config["model"]),
        "training": dict(base_config["training"]),
    }
    config["model"]["model_name"] = "cgr_tmlp"
    config["model"]["spec_output_scale"] = params["spec_output_scale"]
    config["model"]["gamma_init_bias"] = params["gamma_init_bias"]
    config["model"]["topk_ratio"] = params["topk_ratio"]
    config["training"]["train_mode"] = "two_stage"
    config["training"]["base_lr"] = config["training"].get("base_lr", 1e-4)
    config["training"]["spec_lr"] = config["training"].get("spec_lr", 5e-5)
    config["training"]["early_stop_patience"] = config["training"].get("early_stop_patience", 8)
    config["training"]["main_loss_type"] = config["training"].get("main_loss_type", "huber")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def load_prediction(results_dir, model_suffix, dataset):
    prediction_file = results_dir / f"cgr_tmlp{model_suffix}" / dataset / "prediction.json"
    if not prediction_file.exists():
        return {"status": "missing_prediction", "prediction_file": str(prediction_file)}
    with prediction_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics") or {}
    return {
        "status": "ok",
        "prediction_file": str(prediction_file),
        "metric_name": payload.get("metric_name"),
        "metric": payload.get("metric"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "loss": payload.get("loss"),
    }


def save_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / "summary.json"
    csv_file = output_dir / "summary.csv"
    json_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "dataset",
        "tag",
        "status",
        "spec_output_scale",
        "gamma_init_bias",
        "topk_ratio",
        "metric_name",
        "metric",
        "rmse",
        "mae",
        "r2",
        "loss",
        "prediction_file",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {json_file}")
    print(f"Saved summary: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Small key-hyperparameter search for original cgr_tmlp training."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--mode", choices=["default9", "full27"], default="default9")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", default="configs/default/cgr_tmlp.yaml")
    parser.add_argument("--work-dir", default="artifacts/cgr_tmlp_tuning")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    work_dir = Path(args.work_dir)
    config_dir = work_dir / "configs"
    rows = []

    for params in build_grid(args.mode):
        tag = (
            f"__tune_s{tag_float(params['spec_output_scale'])}"
            f"_g{tag_float(params['gamma_init_bias'])}"
            f"_t{tag_float(params['topk_ratio'])}"
        )
        config_file = config_dir / f"cgr_tmlp{tag}.yaml"
        write_config(base_config, params, config_file)
        for dataset in args.datasets:
            cmd = [
                sys.executable,
                "main.py",
                "--model",
                "cgr_tmlp",
                "--dataset",
                dataset,
                "--config",
                str(config_file),
                "--output_suffix",
                tag,
                "--device",
                args.device,
                "--gpu",
                str(args.gpu),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
            ]
            print(" ".join(cmd))
            status = "dry_run"
            if not args.dry_run:
                completed = subprocess.run(cmd)
                status = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"
            row = {
                "dataset": dataset,
                "tag": tag,
                "status": status,
                **params,
            }
            if status == "ok":
                row.update(load_prediction(Path(args.results_dir), tag, dataset))
            rows.append(row)
            save_summary(rows, work_dir)


if __name__ == "__main__":
    main()
