#!/usr/bin/env python3
"""Create immutable RaiderSTREAM/CacheSweep variants without IPC source fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PAIRS = {"raiderstream": "raiderstream_no_ic", "cachesweep": "cachesweep_no_ic"}
INSTRUCTION_NAMES = {"instructions", "instruction", "total_instructions", "total_instruction"}
CYCLE_NAMES = {"cycles", "cycle", "total_cycles", "total_cycle"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def feature_names(info: dict) -> tuple[str, list[str]]:
    for key in ("num_feature_names", "numerical_feature_names", "numeric_feature_names"):
        value = info.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return key, value
    raise ValueError("info.json has no numeric feature-name list; refusing positional deletion")


def canonical(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def equivalent_columns(source: dict[str, np.ndarray], labels: dict[str, np.ndarray], removed: list[int]) -> dict[str, list[dict]]:
    """Detect exact/affine copies only; it intentionally does not infer semantics."""
    full = np.concatenate([source[split] for split in ("train", "val", "test")], axis=0)
    target = np.concatenate([labels[split].reshape(-1) for split in ("train", "val", "test")], axis=0)
    findings = {"exact_removed_copies": [], "affine_removed_copies": [], "exact_label_copies": []}
    for index in range(full.shape[1]):
        if index in removed:
            continue
        candidate = full[:, index]
        if np.array_equal(candidate, target):
            findings["exact_label_copies"].append({"column_index": index})
        for removed_index in removed:
            reference = full[:, removed_index]
            if np.array_equal(candidate, reference):
                findings["exact_removed_copies"].append({"column_index": index, "removed_index": removed_index})
                continue
            spread = float(np.ptp(reference))
            if spread > 0:
                scale = float((candidate[np.argmax(reference)] - candidate[np.argmin(reference)]) / spread)
                offset = float(candidate[np.argmin(reference)] - scale * reference[np.argmin(reference)])
                if np.allclose(candidate, scale * reference + offset, rtol=1e-7, atol=1e-9):
                    findings["affine_removed_copies"].append({"column_index": index, "removed_index": removed_index, "scale": scale, "offset": offset})
    return findings


def clone_dataset(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {target}")
    info_path = source / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    name_key, names = feature_names(info)
    instruction = [index for index, name in enumerate(names) if canonical(name) in INSTRUCTION_NAMES]
    cycles = [index for index, name in enumerate(names) if canonical(name) in CYCLE_NAMES]
    if len(instruction) != 1 or len(cycles) != 1:
        raise ValueError(f"Expected one explicit instructions and cycles field, got {instruction=} {cycles=} in {source}")
    removed = sorted(instruction + cycles)
    arrays, labels, indices = {}, {}, {}
    for split in ("train", "val", "test"):
        arrays[split] = np.load(source / f"X_num_{split}.npy", allow_pickle=False)
        labels[split] = np.load(source / f"y_{split}.npy", allow_pickle=False)
        indices[split] = np.load(source / f"idx_{split}.npy", allow_pickle=False)
        if arrays[split].shape[1] != len(names):
            raise ValueError(f"{source}/{split}: array feature count does not match info.json")
    alternatives = equivalent_columns(arrays, labels, removed)
    if alternatives["exact_removed_copies"] or alternatives["affine_removed_copies"] or alternatives["exact_label_copies"]:
        raise RuntimeError("Found an explicit retained encoding of a removed field or label; inspect provenance before training: " + json.dumps(alternatives))
    keep = [index for index in range(len(names)) if index not in removed]
    target.mkdir(parents=True)
    new_info = dict(info)
    new_info["name"] = target.name
    if isinstance(new_info.get("id"), str):
        new_info["id"] = f"{new_info['id']}--no-ic"
    new_info[name_key] = [names[index] for index in keep]
    for key in ("n_num_features", "n_numerical_features", "num_features"):
        if key in new_info:
            new_info[key] = len(keep)
    (target / "info.json").write_text(json.dumps(new_info, indent=2, ensure_ascii=False), encoding="utf-8")
    checks = []
    for split in ("train", "val", "test"):
        output = arrays[split][:, keep]
        np.save(target / f"X_num_{split}.npy", output)
        np.save(target / f"y_{split}.npy", labels[split])
        np.save(target / f"idx_{split}.npy", indices[split])
        checks.append({"split": split, "before_shape": list(arrays[split].shape), "after_shape": list(output.shape),
                       "label_identical": bool(np.array_equal(labels[split], np.load(target / f"y_{split}.npy", allow_pickle=False))),
                       "index_identical": bool(np.array_equal(indices[split], np.load(target / f"idx_{split}.npy", allow_pickle=False)))})
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(source.resolve()), "target": str(target.resolve()),
                  "removed_indices": removed, "removed_columns": [names[index] for index in removed], "kept_feature_count": len(keep),
                  "source_feature_count": len(names), "split_checks": checks, "equivalence_check": alternatives,
                  "source_hashes": {path.name: digest(path) for path in sorted(source.glob("*.npy")) + [info_path]},
                  "target_hashes": {path.name: digest(path) for path in sorted(target.glob("*.npy")) + [target / "info.json"]}}
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {target}: {len(names)} -> {len(keep)} numerical features; removed {provenance['removed_columns']}")


def register_custom_dataset(root: Path, name: str) -> None:
    """Custom datasets need infos.json for data.env discovery; built-in discovery is directory based."""
    if root.name != "custom_datasets":
        return
    path = root / "infos.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"n_datasets": 0, "regression": 0, "data_list": []}
    entries = info.setdefault("data_list", [])
    if any(entry.get("name") == name for entry in entries if isinstance(entry, dict)):
        return
    entries.append({"name": name, "task_type": "regression"})
    info["n_datasets"] = len(entries)
    info["regression"] = sum(entry.get("task_type") == "regression" for entry in entries if isinstance(entry, dict))
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(PAIRS), default=list(PAIRS))
    args = parser.parse_args()
    for name in args.datasets:
        root = args.data_root / "datasets"
        if not (root / name).is_dir():
            root = args.data_root / "custom_datasets"
        clone_dataset(root / name, root / PAIRS[name])
        register_custom_dataset(root, PAIRS[name])


if __name__ == "__main__":
    main()
