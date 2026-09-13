"""Summarize three-seed results without changing compare_results.py."""

import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = [
    ("ggpl_dynonly_pool_no_channel", "GGPL / graph / no-channel"),
    ("ggpl_dynonly_pool_linear_tokenizer", "linear / graph / channel"),
    ("ggpl_dynonly_pool_no_graph", "GGPL / no-graph / channel"),
    ("ggpl_dynonly_pool_linear_tokenizer_no_channel", "linear / graph / no-channel"),
    ("ggpl_dynonly_pool_no_graph_no_channel", "GGPL / no-graph / no-channel"),
    ("ggpl_dynonly_pool_linear_tokenizer_no_graph", "linear / no-graph / channel"),
    ("ggpl_dynonly_pool_linear_tokenizer_no_graph_no_channel", "linear / no-graph / no-channel"),
    ("ggpl_dynonly_pool", "GGPL / graph / channel (pool)"),
]
DATASETS = {
    "hpcg2": "HPCG",
    "hpgmg3": "HPGMG",
    "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "Mix",
    "raiderstream": "RaiderSTREAM",
    "stream": "STREAM",
    "cachesweep": "CacheSweep",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/dynonly_pool_summary")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def read_metrics(path: Path):
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    metrics = payload.get("metrics") or {}
    return {key: metrics.get(key) for key in ("rmse", "mae", "r2")}


def main():
    args = parse_args()
    root = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, summary = [], []
    for model, label in VARIANTS:
        for dataset, display_name in DATASETS.items():
            rows = []
            for seed in args.seeds:
                path = root / f"{model}_seed{seed}" / dataset / "prediction.json"
                if not path.exists():
                    continue
                row = {
                    "model": model,
                    "scheme": label,
                    "dataset": dataset,
                    "display_dataset": display_name,
                    "seed": seed,
                    **read_metrics(path),
                }
                per_seed.append(row)
                rows.append(row)
            if not rows:
                continue
            aggregate = {
                "model": model,
                "scheme": label,
                "dataset": dataset,
                "display_dataset": display_name,
                "n_seeds": len(rows),
            }
            for metric in ("rmse", "mae", "r2"):
                values = [row[metric] for row in rows if row[metric] is not None]
                aggregate[f"{metric}_mean"] = statistics.mean(values) if values else None
                aggregate[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ) if values else None
            summary.append(aggregate)

    if not per_seed:
        raise FileNotFoundError("No completed pool-ablation prediction files were found")
    for name, rows in (("per_seed", per_seed), ("summary", summary)):
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(per_seed)} seed rows and {len(summary)} aggregate rows to {output_dir}")


if __name__ == "__main__":
    main()
