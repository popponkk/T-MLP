"""Shared read-only helpers for HingeMix efficiency auditing and benchmarks."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - lets empty-result audits still run.
    yaml = None


DATASETS = ("hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161", "raiderstream", "stream", "cachesweep")
DISPLAY_DATASET = {
    "hpcg2": "HPCG", "hpgmg3": "HPGMG", "ramspeed": "RAMspeed",
    "mix_with_five_datasets161": "mix", "raiderstream": "RaiderSTREAM",
    "stream": "STREAM", "cachesweep": "CacheSweep",
}
MODELS = {
    "hingemix": "HingeMix", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "catboost": "CatBoost", "excel-former": "ExcelFormer", "ft-transformer": "FT-Transformer",
    "node": "NODE", "tmlp": "TMLP", "autoint": "AutoInt", "tabm": "TabM",
}
HINGEMIX_FINAL = {"num_breakpoints": 8, "graph_dynamic_rank": 16, "graph_temperature": 16.0, "d_token": 1024, "n_layers": 1}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to inspect saved YAML artifacts; install it in the project environment")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seed_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return match.group(1)
    return "unknown"


def selected_attempt(task_root: Path) -> Path | None:
    marker = read_json(task_root / "parameter_result.json")
    candidate = marker.get("attempt_dir") if marker else None
    if isinstance(candidate, str) and Path(candidate).is_dir():
        return Path(candidate)
    return None


def checkpoint_path(attempt: Path, model: str) -> Path | None:
    candidates = [attempt / "final.pkl"] if model in {"lightgbm", "xgboost", "catboost"} else [attempt / "best.bin", attempt / "final.bin"]
    return next((path for path in candidates if path.is_file()), None)


def configuration_path(attempt: Path) -> Path | None:
    return next((attempt / name for name in ("configs.yaml", "final_config.yaml", "parameter_config.yaml") if (attempt / name).is_file()), None)


def data_config_path(attempt: Path) -> Path | None:
    return next((attempt / name for name in ("data_config.yaml",) if (attempt / name).is_file()), None)


def cache_candidates(data_config: Path | None) -> list[Path]:
    if data_config is None:
        return []
    document = read_yaml(data_config) or {}
    value = document.get("cache_dir")
    directory = Path(value) if isinstance(value, str) else None
    if directory is None or not directory.is_dir():
        return []
    return sorted(directory.glob("cache__*.pickle"))


def config_model_name(path: Path | None) -> str | None:
    document = read_yaml(path) if path else None
    model = (document or {}).get("model")
    name = model.get("model_name") if isinstance(model, dict) else None
    return name if isinstance(name, str) else None


def all_model_files(attempt: Path) -> list[Path]:
    return [path for path in (attempt / "best.bin", attempt / "final.bin", attempt / "final.pkl") if path.is_file()]
