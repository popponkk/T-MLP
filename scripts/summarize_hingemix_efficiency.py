#!/usr/bin/env python3
"""Summarize compatible HingeMix efficiency measurements without changing sources."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hingemix_efficiency_common import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/hingemix_efficiency")
    parser.add_argument("--output-root", default="results/hingemix_efficiency_summary")
    parser.add_argument("--formal-only", action="store_true", help="Include only the formal protocol: warmup=20, rounds=5, requests=100")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], name: str) -> float | None:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


METRICS = (
    "latency_p50_ms",
    "latency_p95_ms",
    "mean_request_latency_ms",
    "throughput_samples_per_second",
    "batch_average_sample_cost_ms",
)
EXPECTED_HINGEMIX_SEEDS = {"0", "1", "2", "3", "4"}


def add_metric_statistics(item: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for metric in METRICS:
        values = [value for row in rows if (value := number(row, f"{metric}_mean" if f"{metric}_mean" in row else metric)) is not None]
        item[f"{metric}_mean"] = float(np.mean(values)) if values else ""
        item[f"{metric}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""


def main() -> None:
    args = parse_args()
    input_root, output_root = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    measurements: list[dict[str, str]] = []
    skipped_legacy: list[str] = []
    skipped_nonformal: list[str] = []
    # ``tasks`` is the resumable layout. ``benchmark_*`` preserves compatibility
    # with short measurements made by the initial development version.
    for path in sorted(list(input_root.glob("tasks/*/summary.csv")) + list(input_root.glob("benchmark_*/summary.csv"))):
        try:
            protocol = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped_legacy.append(f"{path.parent}: missing or invalid protocol.json")
            continue
        if protocol.get("version") != 2:
            skipped_legacy.append(f"{path.parent}: protocol version {protocol.get('version')!r} is not accepted")
            continue
        is_formal = (
            protocol.get("warmup") == 20
            and protocol.get("rounds") == 5
            and protocol.get("requests_per_round") == 100
        )
        if args.formal_only and not is_formal:
            skipped_nonformal.append(str(path.parent))
            continue
        for row in read_rows(path):
            row["measurement_path"] = str(path.parent)
            row["warmup"] = protocol.get("warmup", "")
            row["rounds"] = protocol.get("rounds", "")
            row["requests_per_round"] = protocol.get("requests_per_round", "")
            row["protocol_kind"] = "formal" if is_formal else "smoke_or_custom"
            measurements.append(row)
    stamp = dt.datetime.now().strftime("summary_%Y%m%d_%H%M%S")
    destination = output_root / stamp
    per_fields = sorted({key for row in measurements for key in row}) or ["measurement_path"]
    write_csv(destination / "per_measurement.csv", measurements, per_fields)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("model", "dataset", "seed", "device", "scope", "batch_size", "warmup", "rounds", "requests_per_round", "protocol_fingerprint", "contamination_status")
    for row in measurements:
        if row.get("status") == "measured":
            groups[tuple(row.get(key, "") for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        item = dict(zip(keys, group))
        item["n_measurements"] = len(rows)
        item["status"] = "measured"
        add_metric_statistics(item, rows)
        summaries.append(item)
    fields = sorted({key for row in summaries for key in row}) or list(keys) + ["status"]
    write_csv(destination / "summary.csv", summaries, fields)

    # A timing round is not a training repetition.  Aggregate the already
    # round-aggregated rows across independent training seeds in a separate
    # view, deliberately excluding artifact/protocol fingerprints which differ
    # between checkpoint seeds.
    seed_keys = (
        "model", "dataset", "device", "scope", "batch_size", "warmup",
        "rounds", "requests_per_round", "contamination_status",
    )
    seed_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        seed_groups[tuple(str(row.get(key, "")) for key in seed_keys)].append(row)
    across_seed: list[dict[str, Any]] = []
    for group, rows in sorted(seed_groups.items()):
        item = dict(zip(seed_keys, group))
        observed = [str(row.get("seed", "")) for row in rows]
        unique = set(observed)
        item["training_seeds"] = ",".join(sorted(unique, key=lambda seed: (not seed.isdigit(), seed)))
        item["n_training_seeds"] = len(unique)
        item["n_seed_rows"] = len(rows)
        if len(rows) != len(unique):
            item["status"] = "duplicate_seed_measurements"
        elif unique == EXPECTED_HINGEMIX_SEEDS:
            item["status"] = "complete"
        else:
            item["status"] = "incomplete"
        add_metric_statistics(item, rows)
        across_seed.append(item)
    across_fields = sorted({key for row in across_seed for key in row}) or list(seed_keys) + ["status"]
    write_csv(destination / "across_training_seeds.csv", across_seed, across_fields)
    write_json(destination / "integrity.json", {
        "input_root": str(input_root), "measurement_rows": len(measurements),
        "measured_rows": sum(row.get("status") == "measured" for row in measurements),
        "groups": len(summaries), "across_training_seed_groups": len(across_seed),
        "complete_five_seed_groups": sum(row["status"] == "complete" for row in across_seed),
        "formal_only": args.formal_only,
        "skipped_legacy_or_invalid_measurements": skipped_legacy,
        "skipped_nonformal_measurements": skipped_nonformal,
        "rule": "Only rows with identical model, dataset, training seed, device, scope, batch size, protocol fingerprint, and contamination status are aggregated.",
    })
    (destination / "report.md").write_text(
        "# HingeMix Efficiency Summary\n\n"
        f"- Input root: `{input_root}`\n"
        f"- Measurement rows: {len(measurements)}\n"
        f"- Compatible summary groups: {len(summaries)}\n"
        f"- Across-training-seed groups: {len(across_seed)}\n"
        f"- Complete HingeMix five-seed groups: {sum(row['status'] == 'complete' for row in across_seed)}\n"
        "- This is a timing-round aggregation, not an aggregation of training seeds.\n"
        "- `across_training_seeds.csv` separately summarizes independent training seeds; only exactly seeds 0,1,2,3,4 are marked `complete`.\n"
        "- `model_only` and end-to-end measurements are intentionally never mixed.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summaries)} compatible measurement groups to {destination}")


if __name__ == "__main__":
    main()
