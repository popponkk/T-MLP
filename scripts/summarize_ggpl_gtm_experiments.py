"""Summarize formal GGPL-GTM results without merging historical runs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "Mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}
ABLATIONS = [
    "full", "no_channel", "no_graph", "no_graph_no_channel",
    "linear", "linear_no_channel", "linear_no_graph", "linear_no_graph_no_channel",
]


def prediction_path(results_dir: Path, model: str, ablation: str, dataset: str, seed: int) -> Path:
    root = results_dir / f"{model}_seed{seed}"
    return root / ablation / dataset / "prediction.json" if ablation else root / dataset / "prediction.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/ggpl_gtm_summary")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--include-complete", action="store_true")
    args = parser.parse_args()
    runs = [("ggpl_gtm_ablation", name) for name in ABLATIONS]
    if args.include_complete:
        runs.insert(0, ("ggpl_gtm", ""))
    per_seed, summary = [], []
    for model, ablation in runs:
        for dataset, display_dataset in DATASETS.items():
            rows = []
            for seed in args.seeds:
                path = prediction_path(Path(args.results_dir), model, ablation, dataset, seed)
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as file:
                    metrics = (json.load(file).get("metrics") or {})
                row = {
                    "model": model, "ablation": ablation or None, "dataset": dataset,
                    "display_dataset": display_dataset, "seed": seed,
                    "rmse": metrics.get("rmse"), "mae": metrics.get("mae"), "r2": metrics.get("r2"),
                }
                rows.append(row)
                per_seed.append(row)
            if rows:
                record = {"model": model, "ablation": ablation or None, "dataset": dataset,
                          "display_dataset": display_dataset, "n_seeds": len(rows)}
                for metric in ("rmse", "mae", "r2"):
                    values = [row[metric] for row in rows if row[metric] is not None]
                    record[f"{metric}_mean"] = statistics.mean(values) if values else None
                    record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                summary.append(record)
    if not per_seed:
        raise FileNotFoundError("No completed formal GGPL-GTM results were found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_seed", per_seed), ("summary", summary)):
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(per_seed)} seed rows and {len(summary)} summaries to {output_dir}")


if __name__ == "__main__":
    main()
