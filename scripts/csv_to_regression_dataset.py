import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a regression CSV into the dataset directory format "
            "expected by this project."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
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
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio before validation split. Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting. Default: 42.",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        help="CSV header row index. Use -1 for no header. Default: 0.",
    )
    return parser.parse_args()


def ensure_valid_dataframe(df: pd.DataFrame):
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and one label column.")
    if df.isnull().any().any():
        raise ValueError("CSV contains missing values. Clean them before conversion.")


def build_feature_label_arrays(df: pd.DataFrame, label_col: int):
    if label_col < 0:
        label_col = df.shape[1] + label_col
    if label_col < 0 or label_col >= df.shape[1]:
        raise IndexError(f"label_col out of range: {label_col}")

    feature_df = df.drop(df.columns[label_col], axis=1)
    label_series = df.iloc[:, label_col]

    X_num = feature_df.to_numpy(dtype=np.float32)
    y = label_series.to_numpy(dtype=np.float32)

    feature_names = [str(name) for name in feature_df.columns]
    label_name = str(df.columns[label_col])
    return X_num, y, feature_names, label_name


def split_dataset(X_num: np.ndarray, y: np.ndarray, train_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    indices = np.arange(len(y))
    test_ratio = 1.0 - train_ratio

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": (X_num[train_idx], y[train_idx], train_idx),
        "val": (X_num[val_idx], y[val_idx], val_idx),
        "test": (X_num[test_idx], y[test_idx], test_idx),
    }


def write_dataset(output_dir: Path, splits, feature_names, label_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "name": output_dir.name,
        "id": f"{output_dir.name.lower()}--custom",
        "task_type": "regression",
        "label_name": label_name,
        "n_num_features": len(feature_names),
        "num_feature_names": feature_names,
        "n_cat_features": 0,
        "cat_feature_names": [],
        "train_size": int(len(splits["train"][1])),
        "val_size": int(len(splits["val"][1])),
        "test_size": int(len(splits["test"][1])),
    }

    with open(output_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    for split_name, (X_part, y_part, idx_part) in splits.items():
        np.save(output_dir / f"X_num_{split_name}.npy", X_part.astype(np.float32))
        np.save(output_dir / f"y_{split_name}.npy", y_part.astype(np.float32))
        np.save(output_dir / f"idx_{split_name}.npy", idx_part.astype(np.int64))


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    header = None if args.header == -1 else args.header
    df = pd.read_csv(csv_path, header=header)
    ensure_valid_dataframe(df)

    X_num, y, feature_names, label_name = build_feature_label_arrays(df, args.label_col)
    splits = split_dataset(X_num, y, args.train_ratio, args.seed)

    output_dir = Path("data") / "datasets" / args.name
    write_dataset(output_dir, splits, feature_names, label_name)

    print(f"Created dataset directory: {output_dir}")
    print(
        "Splits:",
        {name: int(len(values[1])) for name, values in splits.items()},
    )


if __name__ == "__main__":
    main()
