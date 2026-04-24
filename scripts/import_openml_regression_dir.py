import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

UNKNOWN_TOKEN = "__unknown__"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_dataset_name(meta: dict, csv_path: Path, mode: str) -> str:
    stem = slugify(csv_path.stem)
    meta_name = slugify(meta.get("name", stem))
    openml_id = meta.get("openml_id")
    if mode == "stem":
        return stem
    if mode == "meta":
        return meta_name
    return f"openml_{openml_id}_{meta_name}" if openml_id is not None else meta_name


def get_column_groups(df: pd.DataFrame, meta: dict) -> Tuple[List[str], List[str], str]:
    target_col = meta["default_target_attribute"]
    feature_names = meta["feature_names"]
    cat_flags = meta["categorical_indicator"]
    if len(feature_names) != len(cat_flags):
        raise ValueError(f"meta mismatch: {len(feature_names)} feature_names vs {len(cat_flags)} categorical flags")
    missing = [c for c in feature_names + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from CSV: {missing[:10]}")

    num_cols: List[str] = []
    cat_cols: List[str] = []
    for col, is_cat in zip(feature_names, cat_flags):
        if is_cat:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols, target_col


def split_indices(n_rows: int, train_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    indices = np.arange(n_rows)
    test_ratio = 1.0 - train_ratio
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=test_ratio, random_state=seed, shuffle=True)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_fill_values(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[dict, dict]:
    num_fill = {}
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        fill = float(series.median()) if not series.dropna().empty else 0.0
        num_fill[col] = fill
    # Use a single fallback token for both missing and unseen categories so
    # validation/test splits never introduce train-unseen category ids.
    cat_fill = {col: UNKNOWN_TOKEN for col in cat_cols}
    return num_fill, cat_fill


def build_train_category_vocab(df: pd.DataFrame, cat_cols: List[str], cat_fill: dict) -> dict:
    vocab = {}
    for col in cat_cols:
        series = df[col].astype("string").fillna(cat_fill[col]).astype(str)
        vocab[col] = set(series.tolist())
    return vocab


def append_unknown_sentinel(train_df: pd.DataFrame, cat_cols: List[str], target_col: str) -> pd.DataFrame:
    if not cat_cols or train_df.empty:
        return train_df
    sentinel = train_df.iloc[[0]].copy()
    for col in cat_cols:
        sentinel[col] = UNKNOWN_TOKEN
    sentinel[target_col] = train_df.iloc[0][target_col]
    return pd.concat([train_df, sentinel], ignore_index=True)


def transform_frame(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    num_fill: dict,
    cat_fill: dict,
    train_vocab: dict | None = None,
):
    x_num = None
    x_cat = None
    if num_cols:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(num_fill)
        x_num = num_df.to_numpy(dtype=np.float32)
    if cat_cols:
        cat_df = df[cat_cols].copy()
        for col in cat_cols:
            series = cat_df[col].astype("string").fillna(cat_fill[col]).astype(str)
            if train_vocab is not None:
                series = series.where(series.isin(train_vocab[col]), cat_fill[col])
            cat_df[col] = series
        x_cat = cat_df.to_numpy(dtype=np.str_)
    return x_num, x_cat


def write_dataset(output_dir: Path, dataset_name: str, meta: dict, num_cols: List[str], cat_cols: List[str], target_col: str, splits: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "name": dataset_name,
        "id": f"{dataset_name}--openml",
        "task_type": "regression",
        "label_name": target_col,
        "n_num_features": len(num_cols),
        "num_feature_names": num_cols,
        "n_cat_features": len(cat_cols),
        "cat_feature_names": cat_cols,
        "train_size": int(len(splits["train"]["y"])),
        "val_size": int(len(splits["val"]["y"])),
        "test_size": int(len(splits["test"]["y"])),
        "normalization": "standard",
        "openml_id": meta.get("openml_id"),
        "source_name": meta.get("name"),
        "original_name": meta.get("original_name"),
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name, payload in splits.items():
        if payload["X_num"] is not None:
            np.save(output_dir / f"X_num_{split_name}.npy", payload["X_num"].astype(np.float32))
        if payload["X_cat"] is not None:
            np.save(output_dir / f"X_cat_{split_name}.npy", payload["X_cat"].astype(np.str_))
        np.save(output_dir / f"y_{split_name}.npy", payload["y"].astype(np.float32))
        np.save(output_dir / f"idx_{split_name}.npy", payload["idx"].astype(np.int64))


def convert_one(csv_path: Path, output_root: Path, train_ratio: float, seed: int, name_mode: str, overwrite: bool) -> Path:
    meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta file for {csv_path.name}: {meta_path}")

    meta = load_meta(meta_path)
    dataset_name = resolve_dataset_name(meta, csv_path, name_mode)
    output_dir = output_root / dataset_name
    if output_dir.exists():
        if not overwrite:
            print(f"[skip] {dataset_name}: output already exists")
            return output_dir
        for pattern in ["info.json", "X_num_*.npy", "X_cat_*.npy", "y_*.npy", "idx_*.npy", "cache__*.pickle"]:
            for path in output_dir.glob(pattern):
                path.unlink()

    df = pd.read_csv(csv_path)
    num_cols, cat_cols, target_col = get_column_groups(df, meta)
    indices = split_indices(len(df), train_ratio=train_ratio, seed=seed)

    raw_train_df = df.iloc[indices["train"]].reset_index(drop=True)
    num_fill, cat_fill = fit_fill_values(raw_train_df, num_cols, cat_cols)
    train_vocab = build_train_category_vocab(raw_train_df, cat_cols, cat_fill)
    train_df = append_unknown_sentinel(raw_train_df, cat_cols, target_col)

    splits = {}
    for split_name, idx in indices.items():
        split_df = train_df if split_name == "train" else df.iloc[idx].reset_index(drop=True)
        X_num, X_cat = transform_frame(
            split_df,
            num_cols,
            cat_cols,
            num_fill,
            cat_fill,
            train_vocab=train_vocab,
        )
        y = pd.to_numeric(split_df[target_col], errors="coerce")
        if y.isnull().any():
            raise ValueError(f"target column contains missing/non-numeric values in {csv_path.name}")
        split_idx = idx if split_name != "train" else np.concatenate([idx, np.array([-1], dtype=np.int64)])
        splits[split_name] = {
            "X_num": X_num,
            "X_cat": X_cat,
            "y": y.to_numpy(dtype=np.float32),
            "idx": split_idx,
        }

    write_dataset(output_dir, dataset_name, meta, num_cols, cat_cols, target_col, splits)
    print(
        f"[done] {dataset_name}: "
        f"num={len(num_cols)}, cat={len(cat_cols)}, "
        f"sizes=({len(indices['train'])}, {len(indices['val'])}, {len(indices['test'])})"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-convert OpenML regression CSV+meta pairs into project dataset format")
    parser.add_argument("--source-dir", required=True, help="Directory containing *.csv and *.meta.json files")
    parser.add_argument("--output-root", default="data/datasets", help="Target root directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio before validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name-mode", choices=["id_name", "meta", "stem"], default="id_name", help="How to build output dataset names")
    parser.add_argument("--only", nargs="*", default=None, help="Optional list of csv stems or output names to convert")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted datasets")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    csv_files = sorted([p for p in source_dir.glob("*.csv") if p.name != "download_summary.csv"])
    if not csv_files:
        raise FileNotFoundError(f"no CSV files found in {source_dir}")

    only = set(args.only or [])
    converted = []
    for csv_path in csv_files:
        meta = load_meta(csv_path.with_suffix(".meta.json"))
        candidate_name = resolve_dataset_name(meta, csv_path, args.name_mode)
        if only and csv_path.stem not in only and candidate_name not in only and meta.get("name") not in only:
            continue
        converted.append(
            convert_one(
                csv_path=csv_path,
                output_root=output_root,
                train_ratio=args.train_ratio,
                seed=args.seed,
                name_mode=args.name_mode,
                overwrite=args.overwrite,
            )
        )

    print(f"converted {len(converted)} dataset(s)")
    for path in converted:
        print(f" - {path}")


if __name__ == "__main__":
    main()
