#!/usr/bin/env python3
"""Summarize compatible HingeMix efficiency measurements without changing sources."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hingemix_efficiency_common import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/hingemix_efficiency")
    parser.add_argument("--output-root", default="results/hingemix_efficiency_summary")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], name: str) -> float | None:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    input_root, output_root = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    measurements: list[dict[str, str]] = []
    # ``tasks`` is the resumable layout. ``benchmark_*`` preserves compatibility
    # with short measurements made by the initial development version.
    for path in sorted(list(input_root.glob("tasks/*/summary.csv")) + list(input_root.glob("benchmark_*/summary.csv"))):
        for row in read_rows(path):
            row["measurement_path"] = str(path.parent)
            measurements.append(row)
    stamp = dt.datetime.now().strftime("summary_%Y%m%d_%H%M%S")
    destination = output_root / stamp
    per_fields = sorted({key for row in measurements for key in row}) or ["measurement_path"]
    write_csv(destination / "per_measurement.csv", measurements, per_fields)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("model", "dataset", "seed", "device", "scope", "batch_size", "protocol_fingerprint", "contamination_status")
    for row in measurements:
        if row.get("status") == "measured":
            groups[tuple(row.get(key, "") for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        item = dict(zip(keys, group))
        item["n_measurements"] = len(rows)
        item["status"] = "measured"
        for metric in ("latency_p50_ms", "latency_p95_ms", "mean_request_latency_ms", "throughput_samples_per_second", "batch_average_sample_cost_ms"):
            values = [value for row in rows if (value := number(row, metric)) is not None]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            item[f"{metric}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
        summaries.append(item)
    fields = sorted({key for row in summaries for key in row}) or list(keys) + ["status"]
    write_csv(destination / "summary.csv", summaries, fields)
    write_json(destination / "integrity.json", {
        "input_root": str(input_root), "measurement_rows": len(measurements),
        "measured_rows": sum(row.get("status") == "measured" for row in measurements),
        "groups": len(summaries),
        "rule": "Only rows with identical model, dataset, training seed, device, scope, batch size, protocol fingerprint, and contamination status are aggregated.",
    })
    (destination / "report.md").write_text(
        "# HingeMix Efficiency Summary\n\n"
        f"- Input root: `{input_root}`\n"
        f"- Measurement rows: {len(measurements)}\n"
        f"- Compatible summary groups: {len(summaries)}\n"
        "- This is a timing-round aggregation, not an aggregation of training seeds.\n"
        "- `model_only` and end-to-end measurements are intentionally never mixed.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summaries)} compatible measurement groups to {destination}")


if __name__ == "__main__":
    main()
