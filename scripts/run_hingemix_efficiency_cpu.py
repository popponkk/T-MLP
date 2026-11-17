#!/usr/bin/env python3
"""Serially measure ready formal baseline artifacts with the CPU protocol.

This is only a queue around benchmark_hingemix_efficiency.py.  It never trains,
alters source experiment results, or treats timing rounds as training seeds.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_efficiency_common import DATASETS, MODELS  # noqa: E402


BASELINES = tuple(name for name in MODELS if name != "hingemix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="Ready-artifact inventory CSV")
    parser.add_argument("--models", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output-root", default="results/hingemix_efficiency")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.warmup < 0 or args.rounds < 1 or args.requests < 1:
        raise SystemExit("threads, rounds, and requests must be positive")
    inventory = read_inventory(Path(args.inventory))
    selected = [
        row for row in inventory
        if row.get("status") == "ready"
        and row.get("model") in args.models
        and row.get("dataset") in args.datasets
    ]
    selected.sort(key=lambda row: (args.models.index(row["model"]), args.datasets.index(row["dataset"]), row.get("seed", "")))
    if not selected:
        raise SystemExit("No ready baseline artifacts match the requested filters")
    print(f"queued={len(selected)} models={','.join(args.models)} datasets={','.join(args.datasets)}", flush=True)
    failures: list[tuple[str, str, str]] = []
    benchmark = REPO_ROOT / "scripts" / "benchmark_hingemix_efficiency.py"
    for row in selected:
        command = [
            sys.executable, str(benchmark), "--inventory", str(Path(args.inventory).resolve()),
            "--model", row["model"], "--dataset", row["dataset"],
            "--device", "cpu", "--threads", str(args.threads),
            "--batch-sizes", *(str(value) for value in args.batch_sizes),
            "--warmup", str(args.warmup), "--rounds", str(args.rounds),
            "--requests", str(args.requests), "--output-root", args.output_root, "--resume",
        ]
        # Historical baseline paths often have no seed directory.  Do not invent
        # one; the benchmark's unfiltered selection is the correct unique record.
        if row.get("seed") not in {"", "unknown"}:
            command.extend(("--seed", row["seed"]))
        print("RUN:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT)
        if completed.returncode:
            failures.append((row["model"], row["dataset"], str(completed.returncode)))
    if failures:
        print("FAILED:", failures, flush=True)
        raise SystemExit(1)
    print(f"completed={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
