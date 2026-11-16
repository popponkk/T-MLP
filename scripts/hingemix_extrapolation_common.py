"""Shared, read-safe helpers for HingeMix configuration-extrapolation scripts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLITS = ("train", "val", "test")
FORMAL_HINGEMIX = {
    "num_breakpoints": 8,
    "graph_dynamic_rank": 16,
    "graph_temperature": 16.0,
    "d_token": 1024,
    "n_layers": 1,
}


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scalar_key(value: Any) -> str:
    """Stable ID/value key without deriving information from row position."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return "<null>"
    return str(value).strip()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not result:
        raise ValueError("experiment name becomes empty after path sanitization")
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def resolve_dataset(repo: Path, name: str) -> Path:
    candidates = [repo / "data" / "datasets" / name, repo / "data" / "custom_datasets" / name]
    found = [path for path in candidates if (path / "info.json").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one dataset directory for {name!r}; found {found}")
    return found[0]


def load_metadata(path: Path, fmt: str | None = None) -> list[dict[str, Any]]:
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if fmt == "json":
        value = read_json(path)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON metadata must be a list of object records")
        return value
    raise ValueError(f"Unsupported metadata format {fmt!r}; use CSV or JSON")


def source_arrays(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    info = read_json(directory / "info.json")
    if not isinstance(info, dict):
        raise ValueError(f"Invalid dataset info: {directory / 'info.json'}")
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {"info.json": sha256_file(directory / "info.json")}
    for split in SPLITS:
        y_path = directory / f"y_{split}.npy"
        x_num_path = directory / f"X_num_{split}.npy"
        idx_path = directory / f"idx_{split}.npy"
        if not y_path.is_file() or not x_num_path.is_file() or not idx_path.is_file():
            raise FileNotFoundError(f"{directory} needs X_num, y and idx arrays for split {split}")
        x_num, y, idx = np.load(x_num_path), np.load(y_path), np.load(idx_path, allow_pickle=True)
        x_cat_path = directory / f"X_cat_{split}.npy"
        x_cat = np.load(x_cat_path, allow_pickle=True) if x_cat_path.is_file() else None
        if len(x_num) != len(y) or len(y) != len(idx) or (x_cat is not None and len(x_cat) != len(y)):
            raise ValueError(f"Inconsistent row counts in {directory} split {split}")
        for path in (x_num_path, y_path, idx_path, x_cat_path):
            if path.is_file():
                hashes[path.name] = sha256_file(path)
        for position, sample_id in enumerate(idx):
            records.append({
                "sample_id": scalar_key(sample_id), "source_split": split, "position": position,
                "x_num": x_num[position], "x_cat": None if x_cat is None else x_cat[position], "y": y[position],
            })
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Saved idx arrays do not provide globally unique sample IDs; cannot safely map metadata")
    return info, records, hashes


def numeric_values(values: list[Any], label: str) -> list[float]:
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain ordered numeric configuration values: {values}") from error
    if not all(np.isfinite(parsed)):
        raise ValueError(f"{label} contains a non-finite configuration value")
    return parsed


def verify_plan(plan: dict[str, Any]) -> None:
    source, extrapolation = plan.get("source"), plan.get("extrapolation")
    if not isinstance(source, dict) or not isinstance(extrapolation, dict):
        raise ValueError("Plan requires source and extrapolation mappings")
    required_source = ("dataset", "metadata_path", "metadata_id_column")
    absent = [key for key in required_source if not source.get(key)]
    if absent:
        raise ValueError(f"source is missing required fields: {absent}")
    required_extrapolation = ("parameter", "train_values", "validation_values", "test_values", "group_fields")
    absent = [key for key in required_extrapolation if not extrapolation.get(key)]
    if absent:
        raise ValueError(f"extrapolation is missing required fields: {absent}")
    groups = list(extrapolation["group_fields"])
    if groups in (["run_id"], ["task_id"], ["sample_id"]):
        raise ValueError("group_fields must describe a full configuration, not only a run/sample ID")
    train = numeric_values(list(extrapolation["train_values"]), "train_values")
    validation = numeric_values(list(extrapolation["validation_values"]), "validation_values")
    test = numeric_values(list(extrapolation["test_values"]), "test_values")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise ValueError("train/validation/test configuration values must be disjoint")
    if min(test) <= max(train + validation):
        raise ValueError("every test configuration value must exceed every development-range value")


def metadata_path(repo: Path, plan: dict[str, Any]) -> Path:
    candidate = Path(str(plan["source"]["metadata_path"]))
    return candidate if candidate.is_absolute() else repo / candidate


def code_evidence(repo: Path) -> dict[str, str]:
    """Small, explicit source fingerprint set recorded with every prepared view."""
    relative = (
        "main.py", "data/processor.py", "utils/data_utils.py", "models/hingemix.py",
        "models/ggpl_tmlp.py", "scripts/prepare_hingemix_extrapolation.py",
        "scripts/run_hingemix_extrapolation.py",
    )
    return {name: sha256_file(repo / name) for name in relative if (repo / name).is_file()}
