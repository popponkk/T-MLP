#!/usr/bin/env python3
"""Create an audited low-configuration -> high-configuration dataset view.

This script never infers run configuration from source-array order.  It requires
an explicit metadata table keyed by the saved dataset ``idx_*.npy`` sample IDs.
Use --inspect first; materialization additionally requires
``source.raw_input_confirmed: true`` in the plan because this repository cannot
prove from an NPY file alone that values were not preprocessed before saving.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hingemix_extrapolation_common import (  # noqa: E402
    SPLITS, code_evidence, fingerprint, load_metadata, load_yaml, metadata_path, numeric_values,
    resolve_dataset, safe_name, scalar_key, source_arrays, verify_plan, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="YAML plan with explicit sample-ID metadata mapping")
    parser.add_argument("--inspect", action="store_true", help="Audit mapping/feasibility only; do not write a dataset view")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-root", default="results/hingemix_extrapolation")
    return parser.parse_args()


def value_matches(value: Any, allowed: list[Any]) -> bool:
    return scalar_key(value) in {scalar_key(item) for item in allowed}


def build_mapping(repo: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    verify_plan(plan)
    source, extra = plan["source"], plan["extrapolation"]
    source_dir = resolve_dataset(repo, str(source["dataset"]))
    info, arrays, hashes = source_arrays(source_dir)
    path = metadata_path(repo, plan)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata table does not exist: {path}")
    metadata = load_metadata(path, source.get("metadata_format"))
    id_column, parameter = str(source["metadata_id_column"]), str(extra["parameter"])
    required = {id_column, parameter, *map(str, extra["group_fields"])}
    for key in ("server_column", "workload_column"):
        if source.get(key):
            required.add(str(source[key]))
    required.update(map(str, (extra.get("fixed_filters") or {}).keys()))
    if not metadata:
        raise ValueError("Metadata table is empty")
    missing = sorted(key for key in required if key not in metadata[0])
    if missing:
        raise ValueError(f"Metadata lacks required columns: {missing}")
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in metadata:
        key = scalar_key(row.get(id_column))
        if key in by_id:
            duplicate_ids.append(key)
        else:
            by_id[key] = row
    if duplicate_ids:
        raise ValueError(f"Metadata sample ID is not unique; examples: {duplicate_ids[:5]}")

    fixed = dict(extra.get("fixed_filters") or {})
    server_column, server_value = source.get("server_column"), source.get("server_value")
    workload_column, workload_value = source.get("workload_column"), source.get("workload_value")
    allowed = {
        "train": list(extra["train_values"]),
        "validation": list(extra["validation_values"]),
        "test": list(extra["test_values"]),
    }
    numeric_allowed = {split: numeric_values(values, f"{split}_values") for split, values in allowed.items()}
    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for sample in arrays:
        row = by_id.get(sample["sample_id"])
        if row is None:
            excluded.append({"sample_id": sample["sample_id"], "reason": "metadata_row_missing"})
            continue
        if server_column and server_value is not None and not value_matches(row[server_column], [server_value]):
            excluded.append({"sample_id": sample["sample_id"], "reason": "server_filter"})
            continue
        if workload_column and workload_value is not None and not value_matches(row[workload_column], [workload_value]):
            excluded.append({"sample_id": sample["sample_id"], "reason": "workload_filter"})
            continue
        rejected_filter = next((name for name, value in fixed.items() if not value_matches(row.get(name), [value])), None)
        if rejected_filter:
            excluded.append({"sample_id": sample["sample_id"], "reason": f"fixed_filter:{rejected_filter}"})
            continue
        try:
            number = float(row[parameter])
        except (TypeError, ValueError):
            excluded.append({"sample_id": sample["sample_id"], "reason": "non_numeric_parameter"})
            continue
        split = next((name for name, values in numeric_allowed.items() if number in values), None)
        if split is None:
            excluded.append({"sample_id": sample["sample_id"], "reason": "parameter_value_not_in_plan"})
            continue
        group_values = [scalar_key(row.get(field)) for field in extra["group_fields"]]
        if any(value in ("", "<null>") for value in group_values):
            excluded.append({"sample_id": sample["sample_id"], "reason": "group_field_missing"})
            continue
        mapped.append({
            **sample, "split": split, "configuration_value": row[parameter],
            "configuration_numeric": number,
            "group_key": "|".join(group_values), "metadata": row,
        })
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for item in mapped:
        group_splits[item["group_key"]].add(item["split"])
    crossing = [key for key, values in group_splits.items() if len(values) > 1]
    if crossing:
        raise ValueError(f"A declared full-configuration group crosses splits; examples: {crossing[:5]}")
    if any(not [item for item in mapped if item["split"] == split] for split in ("train", "validation", "test")):
        counts = collections.Counter(item["split"] for item in mapped)
        raise ValueError(f"At least one extrapolation split is empty after explicit filters: {dict(counts)}")
    evidence = {
        "source_dataset_dir": str(source_dir.resolve()),
        "source_data_hashes": hashes,
        "metadata_path": str(path.resolve()),
        "metadata_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "source_info": info,
        "source_code_sha256": code_evidence(repo),
        "source_array_records": len(arrays),
        "metadata_records": len(metadata),
        "matched_records": len(mapped),
        "excluded_records": len(excluded),
        "excluded_reason_counts": dict(collections.Counter(row["reason"] for row in excluded)),
        "parameter_counts": {
            scalar_key(value): sum(1 for item in mapped if scalar_key(item["configuration_value"]) == scalar_key(value))
            for value in sorted({item["configuration_value"] for item in mapped}, key=lambda item: float(item))
        },
        "split_counts": dict(collections.Counter(item["split"] for item in mapped)),
        "split_group_counts": {split: len({item["group_key"] for item in mapped if item["split"] == split}) for split in ("train", "validation", "test")},
        "other_configuration_distributions": {
            str(field): {
                split: dict(collections.Counter(
                    scalar_key(item["metadata"].get(field)) for item in mapped if item["split"] == split
                ))
                for split in ("train", "validation", "test")
            }
            for field in extra.get("distribution_fields", [])
        },
    }
    return evidence, mapped, excluded


def materialize(repo: Path, plan: dict[str, Any], evidence: dict[str, Any], mapped: list[dict[str, Any]], excluded: list[dict[str, Any]], output_root: Path) -> Path:
    source = plan["source"]
    if source.get("raw_input_confirmed") is not True:
        raise ValueError("Refusing to materialize: source.raw_input_confirmed must be true after verifying source arrays are raw")
    experiment = safe_name(str(plan.get("experiment_id") or f"{source['dataset']}_{plan['extrapolation']['parameter']}_extrapolation"))
    derived = safe_name(str(plan.get("derived_dataset_name") or f"hingemix_extrapolation_{experiment}"))
    identity = {"plan": plan, "evidence": evidence, "mapped": [
        {key: item[key] for key in ("sample_id", "split", "configuration_value", "group_key", "source_split", "position")}
        for item in mapped
    ]}
    manifest_fingerprint = fingerprint(identity)
    manifest_dir = output_root / "manifests" / experiment
    view_dir = repo / "data" / "datasets" / derived
    existing = (manifest_dir / "manifest.json")
    if existing.is_file():
        prior = __import__("json").loads(existing.read_text(encoding="utf-8"))
        if prior.get("manifest_fingerprint") != manifest_fingerprint:
            raise FileExistsError(f"Existing manifest conflicts with this plan: {existing}")
        if not (view_dir / "info.json").is_file():
            raise FileNotFoundError(f"Manifest exists but derived dataset view is absent: {view_dir}")
        return manifest_dir
    if view_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing derived dataset directory: {view_dir}")
    grouped = {"train": [], "val": [], "test": []}
    aliases = {"train": "train", "validation": "val", "test": "test"}
    for item in mapped:
        grouped[aliases[item["split"]]].append(item)
    view_dir.mkdir(parents=True)
    try:
        for split in SPLITS:
            rows = grouped[split]
            np.save(view_dir / f"X_num_{split}.npy", np.stack([row["x_num"] for row in rows]))
            if rows[0]["x_cat"] is not None:
                np.save(view_dir / f"X_cat_{split}.npy", np.stack([row["x_cat"] for row in rows]))
            np.save(view_dir / f"y_{split}.npy", np.asarray([row["y"] for row in rows]))
            np.save(view_dir / f"idx_{split}.npy", np.asarray([row["sample_id"] for row in rows], dtype=str))
        info = dict(evidence["source_info"])
        info.update({
            "name": derived, "extrapolation_experiment": experiment,
            "extrapolation_manifest_fingerprint": manifest_fingerprint,
            "extrapolation_source_dataset": source["dataset"],
            "extrapolation_parameter": plan["extrapolation"]["parameter"],
        })
        write_json(view_dir / "info.json", info)
    except Exception:
        shutil.rmtree(view_dir, ignore_errors=True)
        raise
    manifest = {
        "schema_version": 1, "experiment_id": experiment, "derived_dataset_name": derived,
        "derived_dataset_dir": str(view_dir.resolve()), "manifest_fingerprint": manifest_fingerprint,
        "plan": plan, "evidence": evidence,
        "rows": {split: len(rows) for split, rows in grouped.items()},
        "groups": {split: len({row['group_key'] for row in rows}) for split, rows in grouped.items()},
    }
    write_json(manifest_dir / "manifest.json", manifest)
    sample_rows = []
    for item in mapped:
        sample_rows.append({
            "sample_id": item["sample_id"], "source_split": item["source_split"], "source_position": item["position"],
            "split": item["split"], "group_key": item["group_key"], "configuration_value": item["configuration_value"],
            "server": item["metadata"].get(source.get("server_column", ""), ""),
            "workload": item["metadata"].get(source.get("workload_column", ""), ""),
        })
    write_csv(manifest_dir / "samples.csv", sample_rows, list(sample_rows[0]))
    write_csv(manifest_dir / "excluded_records.csv", excluded, ["sample_id", "reason"])
    return manifest_dir


def main() -> None:
    args = parse_args()
    repo = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan).resolve()
    plan = load_yaml(plan_path)
    evidence, mapped, excluded = build_mapping(repo, plan)
    experiment = safe_name(str(plan.get("experiment_id") or f"{plan['source']['dataset']}_{plan['extrapolation']['parameter']}_extrapolation"))
    output_root = Path(args.output_root).resolve()
    audit_dir = output_root / "manifests" / experiment
    write_json(audit_dir / "feasibility.json", evidence)
    write_csv(audit_dir / "feasibility.csv", [{
        "dataset": plan["source"]["dataset"], "server": plan["source"].get("server_value", ""),
        "workload": plan["source"].get("workload_value", ""), "parameter": plan["extrapolation"]["parameter"],
        "parameter_counts": __import__("json").dumps(evidence["parameter_counts"], ensure_ascii=False),
        "mapping_source": evidence["metadata_path"], "matched_records": evidence["matched_records"],
        "excluded_records": evidence["excluded_records"], "can_construct": True,
    }], ["dataset", "server", "workload", "parameter", "parameter_counts", "mapping_source", "matched_records", "excluded_records", "can_construct"])
    if args.inspect:
        print(f"Feasibility audit written to {audit_dir}")
        print(f"matched={len(mapped)} excluded={len(excluded)} split_counts={evidence['split_counts']}")
        return
    manifest_dir = materialize(repo, plan, evidence, mapped, excluded, output_root)
    print(f"Prepared extrapolation dataset and manifest: {manifest_dir}")


if __name__ == "__main__":
    main()
