import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build leave-one-dataset-out regression datasets from a set of source CSVs. "
            "For each held-out dataset, the remaining datasets are concatenated for train/val, "
            "and the held-out dataset is used entirely as test."
        )
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing source CSV files such as hpcg2.csv, hpgmg3.csv, etc.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset base names to use, for example: hpcg2 hpgmg3 ramspeed raiderstream_IPC stream_IPC",
    )
    parser.add_argument(
        "--output-root",
        default="data/datasets",
        help="Output root for generated dataset directories. Default: data/datasets",
    )
    parser.add_argument(
        "--name-prefix",
        default="lood",
        help="Prefix for generated dataset names. Default: lood",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=-1,
        help="Label column index in each CSV. Default: last column.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio split from the concatenated training pool. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting. Default: 42",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        help="CSV header row index. Use -1 for no header. Default: 0",
    )
    parser.add_argument(
        "--drop-cols",
        default="",
        help="Comma-separated feature columns to drop before alignment.",
    )
    parser.add_argument(
        "--keep-shared-only",
        action="store_true",
        help="Keep only shared feature columns across datasets. Otherwise raise on mismatch.",
    )
    return parser.parse_args()


def parse_drop_cols(drop_cols: str) -> List[str]:
    if not drop_cols:
        return []
    return [c.strip() for c in drop_cols.split(",") if c.strip()]


def resolve_label_col(df: pd.DataFrame, label_col: int) -> int:
    if label_col < 0:
        label_col = df.shape[1] + label_col
    if label_col < 0 or label_col >= df.shape[1]:
        raise IndexError(f"label_col out of range: {label_col}")
    return label_col


def ensure_valid_dataframe(df: pd.DataFrame, name: str):
    if df.shape[1] < 2:
        raise ValueError(f"{name}: CSV must contain at least one feature column and one label column.")
    if df.isnull().any().any():
        raise ValueError(f"{name}: CSV contains missing values. Clean them before conversion.")


def find_csv_path(source_dir: Path, dataset_name: str) -> Path:
    candidates = [
        source_dir / f"{dataset_name}.csv",
        source_dir / f"{dataset_name}.CSV",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find CSV for dataset '{dataset_name}' under {source_dir}. "
        f"Expected one of: {candidates[0].name}, {candidates[1].name}"
    )


def split_feature_label_df(
    df: pd.DataFrame, label_col: int
) -> Tuple[pd.DataFrame, pd.Series, str]:
    label_col = resolve_label_col(df, label_col)
    label_name = str(df.columns[label_col])
    feature_df = df.drop(df.columns[label_col], axis=1)
    label_series = df.iloc[:, label_col]
    return feature_df, label_series, label_name


def load_source_datasets(
    source_dir: Path,
    dataset_names: List[str],
    label_col: int,
    header: int,
    drop_cols: List[str],
) -> Dict[str, Dict[str, object]]:
    header_arg = None if header == -1 else header
    loaded: Dict[str, Dict[str, object]] = {}

    for dataset_name in dataset_names:
        csv_path = find_csv_path(source_dir, dataset_name)
        df = pd.read_csv(csv_path, header=header_arg)
        ensure_valid_dataframe(df, dataset_name)
        feature_df, label_series, label_name = split_feature_label_df(df, label_col)
        if drop_cols:
            feature_df = feature_df.drop(
                columns=[c for c in drop_cols if c in feature_df.columns]
            )
        loaded[dataset_name] = {
            "csv_path": csv_path,
            "features": feature_df,
            "target": label_series,
            "label_name": label_name,
        }
    return loaded


def align_feature_columns(
    datasets: Dict[str, Dict[str, object]], keep_shared_only: bool
) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    dataset_names = list(datasets.keys())
    base_name = dataset_names[0]
    base_cols = [str(c) for c in datasets[base_name]["features"].columns]
    common_cols = set(base_cols)
    all_cols = set(base_cols)

    for dataset_name in dataset_names[1:]:
        cols = [str(c) for c in datasets[dataset_name]["features"].columns]
        common_cols &= set(cols)
        all_cols |= set(cols)

    if not keep_shared_only:
        mismatched = []
        for dataset_name in dataset_names[1:]:
            cols = [str(c) for c in datasets[dataset_name]["features"].columns]
            if cols != base_cols:
                mismatched.append(dataset_name)
        if mismatched:
            raise ValueError(
                "Feature columns differ across datasets. "
                "Re-run with --keep-shared-only to keep only shared columns."
            )

    ordered_common_cols = [c for c in base_cols if c in common_cols]
    dropped = sorted(all_cols - common_cols)
    if not ordered_common_cols:
        raise ValueError("No shared feature columns remain after alignment.")

    aligned: Dict[str, Dict[str, object]] = {}
    for dataset_name, payload in datasets.items():
        feature_df = payload["features"].loc[:, ordered_common_cols].copy()
        aligned[dataset_name] = {
            **payload,
            "features": feature_df,
        }
    return aligned, ordered_common_cols, dropped


def build_arrays(feature_df: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    X_num = feature_df.to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=np.float32)
    return X_num, y


def split_train_val(X_num: np.ndarray, y: np.ndarray, val_ratio: float, seed: int):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return {
        "train": (X_num[train_idx], y[train_idx], train_idx),
        "val": (X_num[val_idx], y[val_idx], val_idx),
    }


def write_dataset(
    output_dir: Path,
    splits,
    feature_names: List[str],
    label_name: str,
    source_train: List[str],
    source_test: str,
    metadata: Dict[str, object],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "name": output_dir.name,
        "id": f"{output_dir.name.lower()}--leave-one-dataset-out",
        "task_type": "regression",
        "normalization": "standard",
        "label_name": label_name,
        "n_num_features": len(feature_names),
        "num_feature_names": feature_names,
        "n_cat_features": 0,
        "cat_feature_names": [],
        "train_size": int(len(splits["train"][1])),
        "val_size": int(len(splits["val"][1])),
        "test_size": int(len(splits["test"][1])),
        "split_mode": "leave_one_dataset_out",
        "source_train_datasets": source_train,
        "source_test_dataset": source_test,
        **metadata,
    }

    with open(output_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    for split_name, (X_part, y_part, idx_part) in splits.items():
        np.save(output_dir / f"X_num_{split_name}.npy", X_part.astype(np.float32))
        np.save(output_dir / f"y_{split_name}.npy", y_part.astype(np.float32))
        np.save(output_dir / f"idx_{split_name}.npy", idx_part.astype(np.int64))


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    dataset_names = args.datasets
    if len(dataset_names) < 2:
        raise ValueError("At least two datasets are required for leave-one-dataset-out.")

    loaded = load_source_datasets(
        source_dir=source_dir,
        dataset_names=dataset_names,
        label_col=args.label_col,
        header=args.header,
        drop_cols=parse_drop_cols(args.drop_cols),
    )

    label_names = {payload["label_name"] for payload in loaded.values()}
    if len(label_names) != 1:
        raise ValueError(f"Label names differ across datasets: {sorted(label_names)}")

    aligned, feature_names, dropped_nonshared = align_feature_columns(
        loaded, keep_shared_only=args.keep_shared_only
    )

    output_root = Path(args.output_root)
    label_name = next(iter(label_names))

    for heldout_name in dataset_names:
        train_names = [name for name in dataset_names if name != heldout_name]

        train_frames = [aligned[name]["features"] for name in train_names]
        train_targets = [aligned[name]["target"] for name in train_names]
        train_concat = pd.concat(train_frames, axis=0, ignore_index=True)
        target_concat = pd.concat(train_targets, axis=0, ignore_index=True)

        X_trainval, y_trainval = build_arrays(train_concat, target_concat)
        splits = split_train_val(X_trainval, y_trainval, args.val_ratio, args.seed)

        test_features = aligned[heldout_name]["features"]
        test_target = aligned[heldout_name]["target"]
        X_test, y_test = build_arrays(test_features, test_target)
        test_idx = np.arange(len(y_test), dtype=np.int64)
        splits["test"] = (X_test, y_test, test_idx)

        output_name = f"{args.name_prefix}_{heldout_name}"
        metadata = {
            "source_csv_dir": str(source_dir),
            "source_csv_files": {
                name: str(aligned[name]["csv_path"]) for name in dataset_names
            },
            "dropped_nonshared_columns": dropped_nonshared,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        }
        write_dataset(
            output_dir=output_root / output_name,
            splits=splits,
            feature_names=feature_names,
            label_name=label_name,
            source_train=train_names,
            source_test=heldout_name,
            metadata=metadata,
        )
        print(
            f"[done] {output_name}: "
            f"train={len(splits['train'][1])}, "
            f"val={len(splits['val'][1])}, "
            f"test={len(splits['test'][1])}"
        )


if __name__ == "__main__":
    main()
