import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert two regression CSVs into the dataset directory format "
            "expected by this project for cross-workload evaluation. "
            "Train/val come from --train-csv only; test comes entirely from --test-csv."
        )
    )
    parser.add_argument("--train-csv", required=True, help="Path to the training-workload CSV.")
    parser.add_argument("--test-csv", required=True, help="Path to the unseen-workload CSV used as test set.")
    parser.add_argument(
        "--name",
        required=True,
        help="Output dataset name. The result will be written to data/datasets/<name>.",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=-1,
        help="Label column index. Default: last column.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio taken from the training workload only. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting. Default: 42.",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        help="CSV header row index. Use -1 for no header. Default: 0.",
    )
    parser.add_argument(
        "--drop-cols",
        default="",
        help=(
            "Comma-separated feature columns to drop from both CSVs before building the dataset, "
            "for example: instructions,cycles"
        ),
    )
    parser.add_argument(
        "--keep-shared-only",
        action="store_true",
        help="Keep only shared feature columns if the two CSVs differ. Default: False (raise error on mismatch).",
    )
    parser.add_argument(
        "--drop-constant-cols",
        action="store_true",
        help="Drop columns that are constant in the training workload.",
    )
    parser.add_argument(
        "--drop-duplicate-cols",
        action="store_true",
        help="Drop duplicate feature columns detected in the training workload.",
    )
    return parser.parse_args()


def ensure_valid_dataframe(df: pd.DataFrame, name: str):
    if df.shape[1] < 2:
        raise ValueError(f"{name}: CSV must contain at least one feature column and one label column.")
    if df.isnull().any().any():
        raise ValueError(f"{name}: CSV contains missing values. Clean them before conversion.")


def resolve_label_col(df: pd.DataFrame, label_col: int) -> int:
    if label_col < 0:
        label_col = df.shape[1] + label_col
    if label_col < 0 or label_col >= df.shape[1]:
        raise IndexError(f"label_col out of range: {label_col}")
    return label_col


def split_feature_label_df(df: pd.DataFrame, label_col: int) -> Tuple[pd.DataFrame, pd.Series, str]:
    label_col = resolve_label_col(df, label_col)
    label_name = str(df.columns[label_col])
    feature_df = df.drop(df.columns[label_col], axis=1)
    label_series = df.iloc[:, label_col]
    return feature_df, label_series, label_name


def parse_drop_cols(drop_cols: str) -> List[str]:
    if not drop_cols:
        return []
    return [c.strip() for c in drop_cols.split(",") if c.strip()]


def align_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keep_shared_only: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_cols = [str(c) for c in train_df.columns]
    test_cols = [str(c) for c in test_df.columns]

    if train_cols == test_cols:
        return train_df.copy(), test_df.copy(), []

    train_set = set(train_cols)
    test_set = set(test_cols)
    only_train = [c for c in train_cols if c not in test_set]
    only_test = [c for c in test_cols if c not in train_set]

    if not keep_shared_only:
        raise ValueError(
            "Train/test feature columns do not match. "
            f"Only in train: {only_train[:10]}{'...' if len(only_train) > 10 else ''}; "
            f"only in test: {only_test[:10]}{'...' if len(only_test) > 10 else ''}. "
            "Use --keep-shared-only to retain only shared columns."
        )

    shared = [c for c in train_cols if c in test_set]
    dropped = only_train + only_test
    return train_df.loc[:, shared].copy(), test_df.loc[:, shared].copy(), dropped


def drop_requested_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    removed = [c for c in drop_cols if c in train_df.columns or c in test_df.columns]
    if not removed:
        return train_df, test_df, []

    train_df = train_df.drop(columns=[c for c in removed if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in removed if c in test_df.columns])
    return train_df, test_df, removed


def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    nunique = df.nunique(dropna=False)
    return [str(c) for c in nunique.index[nunique <= 1]]


def detect_duplicate_columns(df: pd.DataFrame) -> List[str]:
    duplicates: List[str] = []
    seen = {}
    for col in df.columns:
        key = tuple(df[col].tolist())
        if key in seen:
            duplicates.append(str(col))
        else:
            seen[key] = str(col)
    return duplicates


def build_arrays(feature_df: pd.DataFrame, label_series: pd.Series):
    X_num = feature_df.to_numpy(dtype=np.float32)
    y = label_series.to_numpy(dtype=np.float32)
    feature_names = [str(name) for name in feature_df.columns]
    return X_num, y, feature_names


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


def write_dataset(output_dir: Path, splits, feature_names, label_name, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "name": output_dir.name,
        "id": f"{output_dir.name.lower()}--cross-workload-custom",
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
        "split_mode": "cross_workload",
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
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV file not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV file not found: {test_csv}")

    header = None if args.header == -1 else args.header
    train_raw = pd.read_csv(train_csv, header=header)
    test_raw = pd.read_csv(test_csv, header=header)
    ensure_valid_dataframe(train_raw, "train")
    ensure_valid_dataframe(test_raw, "test")

    train_features, train_y_series, train_label = split_feature_label_df(train_raw, args.label_col)
    test_features, test_y_series, test_label = split_feature_label_df(test_raw, args.label_col)
    if train_label != test_label:
        raise ValueError(f"Label name mismatch: train={train_label}, test={test_label}")

    train_features, test_features, dropped_nonshared = align_feature_columns(
        train_features, test_features, args.keep_shared_only
    )

    train_features, test_features, dropped_requested = drop_requested_columns(
        train_features, test_features, parse_drop_cols(args.drop_cols)
    )

    dropped_constants: List[str] = []
    if args.drop_constant_cols:
        dropped_constants = detect_constant_columns(train_features)
        if dropped_constants:
            train_features = train_features.drop(columns=dropped_constants)
            test_features = test_features.drop(columns=[c for c in dropped_constants if c in test_features.columns])

    dropped_duplicates: List[str] = []
    if args.drop_duplicate_cols:
        dropped_duplicates = detect_duplicate_columns(train_features)
        if dropped_duplicates:
            train_features = train_features.drop(columns=dropped_duplicates)
            test_features = test_features.drop(columns=[c for c in dropped_duplicates if c in test_features.columns])

    if train_features.empty:
        raise ValueError("No feature columns remain after filtering.")

    X_trainval, y_trainval, feature_names = build_arrays(train_features, train_y_series)
    X_test, y_test, _ = build_arrays(test_features.loc[:, feature_names], test_y_series)

    splits = split_train_val(X_trainval, y_trainval, args.val_ratio, args.seed)
    test_idx = np.arange(len(y_test), dtype=np.int64)
    splits["test"] = (X_test, y_test, test_idx)

    metadata = {
        "source_train_csv": str(train_csv),
        "source_test_csv": str(test_csv),
        "dropped_nonshared_columns": dropped_nonshared,
        "dropped_requested_columns": dropped_requested,
        "dropped_constant_columns": dropped_constants,
        "dropped_duplicate_columns": dropped_duplicates,
    }

    output_dir = Path("data") / "datasets" / args.name
    write_dataset(output_dir, splits, feature_names, train_label, metadata)

    print(f"Created cross-workload dataset directory: {output_dir}")
    print("Feature count:", len(feature_names))
    print(
        "Splits:",
        {name: int(len(values[1])) for name, values in splits.items()},
    )
    print("Dropped nonshared columns:", len(dropped_nonshared))
    print("Dropped requested columns:", len(dropped_requested))
    print("Dropped constant columns:", len(dropped_constants))
    print("Dropped duplicate columns:", len(dropped_duplicates))


if __name__ == "__main__":
    main()
