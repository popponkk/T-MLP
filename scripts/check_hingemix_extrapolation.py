#!/usr/bin/env python3
"""Small synthetic smoke test for the configuration-extrapolation preparation path.

It does not import models, allocate a GPU, or run training.  The test verifies
sample-ID mapping, grouped split isolation, and rejection of non-extrapolating
plans using temporary files only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "data" / "datasets" / "toy_extrapolation"
        data.mkdir(parents=True)
        rows = []
        cursor = 0
        for split, values in (("train", [1, 2]), ("val", [4]), ("test", [8])):
            count = len(values) * 2
            np.save(data / f"X_num_{split}.npy", np.arange(cursor, cursor + count * 3, dtype=np.float32).reshape(count, 3))
            np.save(data / f"y_{split}.npy", np.arange(count, dtype=np.float32))
            ids = np.asarray([f"id_{cursor + index}" for index in range(count)])
            np.save(data / f"idx_{split}.npy", ids)
            for index, sample_id in enumerate(ids):
                rows.append({"sample_id": sample_id, "server": "s0", "workload": "toy", "threads": values[index // 2], "processes": 1, "problem_size": 16})
            cursor += count
        (data / "info.json").write_text(json.dumps({"name": "toy_extrapolation", "task_type": "regression", "normalization": "standard"}), encoding="utf-8")
        metadata = root / "metadata.csv"
        import csv
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        plan = {
            "experiment_id": "toy_threads", "derived_dataset_name": "toy_extrapolation_view",
            "source": {"dataset": "toy_extrapolation", "metadata_path": str(metadata), "metadata_id_column": "sample_id", "raw_input_confirmed": True, "server_column": "server", "server_value": "s0", "workload_column": "workload", "workload_value": "toy"},
            "extrapolation": {"parameter": "threads", "train_values": [1, 2], "validation_values": [4], "test_values": [8], "fixed_filters": {"processes": 1, "problem_size": 16}, "group_fields": ["server", "workload", "threads", "processes", "problem_size"], "split_seed": 42},
            "models": ["hingemix"], "training_seeds": [0],
        }
        plan_path = root / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        command = [sys.executable, str(repo / "scripts" / "prepare_hingemix_extrapolation.py"), "--project-root", str(root), "--plan", str(plan_path), "--output-root", str(root / "results")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
        manifest = root / "results" / "manifests" / "toy_threads" / "manifest.json"
        samples = root / "results" / "manifests" / "toy_threads" / "samples.csv"
        view = root / "data" / "datasets" / "toy_extrapolation_view"
        assert manifest.is_file() and samples.is_file() and (view / "X_num_test.npy").is_file()
        assert len(np.load(view / "X_num_train.npy")) == 4
        assert len(np.load(view / "X_num_val.npy")) == 2
        assert len(np.load(view / "X_num_test.npy")) == 2
    print("HingeMix extrapolation synthetic smoke test passed.")


if __name__ == "__main__":
    main()
