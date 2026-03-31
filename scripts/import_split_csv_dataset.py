import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SPLITS = ["train", "val", "test"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_column_groups(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for column in df.columns:
        if column.startswith("num__"):
            num_cols.append(column)
        elif column.startswith("cat__"):
            cat_cols.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            num_cols.append(column)
        else:
            cat_cols.append(column)
    return num_cols, cat_cols


def read_split(source_dir: Path, split: str) -> tuple[pd.DataFrame, pd.Series, Optional[dict]]:
    split_dir = source_dir / split
    features_path = split_dir / "features.csv"
    target_path = split_dir / "target.csv"
    meta_path = split_dir / "meta.json"

    if not features_path.exists():
        raise FileNotFoundError(f"missing features file: {features_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"missing target file: {target_path}")

    features = pd.read_csv(features_path)
    target = pd.read_csv(target_path)
    if target.shape[1] != 1:
        raise ValueError(f"{target_path} must contain exactly one target column")
    if len(features) != len(target):
        raise ValueError(
            f"row count mismatch in split '{split}': "
            f"{len(features)} features vs {len(target)} targets"
        )

    meta = load_json(meta_path) if meta_path.exists() else None
    return features, target.iloc[:, 0], meta


def build_dataset(
    source_dir: Path,
    dataset_name: str,
    output_root: Path,
    task_type: Optional[str] = None,
) -> Path:
    split_frames: Dict[str, pd.DataFrame] = {}
    split_targets: Dict[str, pd.Series] = {}
    split_meta: Dict[str, Optional[dict]] = {}

    for split in SPLITS:
        features, target, meta = read_split(source_dir, split)
        split_frames[split] = features
        split_targets[split] = target
        split_meta[split] = meta

    train_meta = split_meta["train"] or {}
    inferred_task = task_type or train_meta.get("task") or "regression"
    if inferred_task != "regression":
        raise NotImplementedError(
            f"this importer currently supports regression only, got task={inferred_task!r}"
        )

    train_columns = split_frames["train"].columns.tolist()
    for split in SPLITS[1:]:
        if split_frames[split].columns.tolist() != train_columns:
            raise ValueError(f"column mismatch between train and {split}")

    num_cols, cat_cols = infer_column_groups(split_frames["train"])

    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        frame = split_frames[split]
        target = split_targets[split].to_numpy(dtype=np.float32)

        if num_cols:
            x_num = frame[num_cols].to_numpy(dtype=np.float32)
            np.save(output_dir / f"X_num_{split}.npy", x_num)

        if cat_cols:
            x_cat = frame[cat_cols].astype(str).to_numpy(dtype=np.str_)
            np.save(output_dir / f"X_cat_{split}.npy", x_cat)

        np.save(output_dir / f"y_{split}.npy", target)
        np.save(output_dir / f"idx_{split}.npy", np.arange(len(frame), dtype=np.int64))

    info = {
        "name": dataset_name,
        "id": dataset_name,
        "task_type": inferred_task,
        "label_name": split_targets["train"].name or "target",
        "n_num_features": len(num_cols),
        "num_feature_names": num_cols,
        "n_cat_features": len(cat_cols),
        "cat_feature_names": cat_cols,
        "train_size": len(split_frames["train"]),
        "val_size": len(split_frames["val"]),
        "test_size": len(split_frames["test"]),
        "normalization": "standard",
    }
    (output_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a split CSV dataset into data/datasets/<name>/ format"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing train/ val/ test subdirectories",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name to create under data/datasets/",
    )
    parser.add_argument(
        "--output-root",
        default="data/datasets",
        help="Root directory where converted dataset will be written",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Override task type, defaults to meta.json task or regression",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    output_dir = build_dataset(
        source_dir=source_dir,
        dataset_name=args.name,
        output_root=output_root,
        task_type=args.task,
    )
    print(f"converted dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
