"""Aggregate completed independent-``nn.Linear`` ablations across seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = [
    ("ggpl_dynonly_ablation_independent_nnlinear", "independent-nn.Linear / graph / channel"),
    ("ggpl_dynonly_ablation_independent_nnlinear_no_channel", "independent-nn.Linear / graph / no-channel"),
    ("ggpl_dynonly_ablation_independent_nnlinear_no_graph", "independent-nn.Linear / no-graph / channel"),
    ("ggpl_dynonly_ablation_independent_nnlinear_no_graph_no_channel", "independent-nn.Linear / no-graph / no-channel"),
]
DATASETS = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "Mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir", default="results/dynonly_independent_nnlinear_summary"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    per_seed, summary = [], []
    for model, scheme in VARIANTS:
        for dataset, display_dataset in DATASETS.items():
            rows = []
            for seed in args.seeds:
                path = (
                    Path(args.results_dir)
                    / f"{model}_seed{seed}"
                    / dataset
                    / "prediction.json"
                )
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as file:
                    metrics = (json.load(file).get("metrics") or {})
                row = {
                    "model": model, "scheme": scheme, "dataset": dataset,
                    "display_dataset": display_dataset, "seed": seed,
                    "rmse": metrics.get("rmse"), "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                }
                per_seed.append(row)
                rows.append(row)
            if not rows:
                continue
            result = {
                "model": model, "scheme": scheme, "dataset": dataset,
                "display_dataset": display_dataset, "n_seeds": len(rows),
            }
            for metric in ("rmse", "mae", "r2"):
                values = [row[metric] for row in rows if row[metric] is not None]
                result[f"{metric}_mean"] = statistics.mean(values) if values else None
                result[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            summary.append(result)
    if not per_seed:
        raise FileNotFoundError("No completed independent-nn.Linear ablation results were found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_seed", per_seed), ("summary", summary)):
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(per_seed)} seed rows and {len(summary)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
