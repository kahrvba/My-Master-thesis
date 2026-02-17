#!/usr/bin/env python3
"""
Assess dataset quality and split balance for training readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess synthetic dataset quality.")
    parser.add_argument("--index", default="My-Master-thesis/data/synthetic/index.csv")
    parser.add_argument("--splits-dir", default="My-Master-thesis/data/synthetic/splits")
    parser.add_argument("--output", default="My-Master-thesis/artifacts/dataset_quality_report.json")
    args = parser.parse_args()

    index_df = pd.read_csv(args.index)
    train_df = pd.read_csv(Path(args.splits_dir) / "train_ids.csv")
    val_df = pd.read_csv(Path(args.splits_dir) / "val_ids.csv")
    test_df = pd.read_csv(Path(args.splits_dir) / "test_ids.csv")

    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])
    index_ids = set(index_df["sample_id"])

    group_cols = ["size_bucket", "distribution_tag", "structure_tag"]
    grouped = index_df.groupby(group_cols).size()

    report = {
        "counts": {
            "index": int(len(index_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_integrity": {
            "train_val_overlap": int(len(train_ids & val_ids)),
            "train_test_overlap": int(len(train_ids & test_ids)),
            "val_test_overlap": int(len(val_ids & test_ids)),
            "missing_from_splits": int(len(index_ids - train_ids - val_ids - test_ids)),
            "unknown_ids_in_splits": int(len((train_ids | val_ids | test_ids) - index_ids)),
        },
        "balance": {
            "size_counts": index_df["size_bucket"].value_counts().sort_index().to_dict(),
            "distribution_counts": index_df["distribution_tag"].value_counts().sort_index().to_dict(),
            "structure_counts": index_df["structure_tag"].value_counts().sort_index().to_dict(),
            "stratified_combo_min": int(grouped.min()),
            "stratified_combo_max": int(grouped.max()),
            "stratified_combo_unique_counts": sorted([int(x) for x in grouped.unique()]),
        },
    }

    checks = []
    checks.append(report["split_integrity"]["train_val_overlap"] == 0)
    checks.append(report["split_integrity"]["train_test_overlap"] == 0)
    checks.append(report["split_integrity"]["val_test_overlap"] == 0)
    checks.append(report["split_integrity"]["missing_from_splits"] == 0)
    checks.append(report["split_integrity"]["unknown_ids_in_splits"] == 0)
    checks.append(report["balance"]["stratified_combo_min"] == report["balance"]["stratified_combo_max"])

    report["ready_for_training"] = bool(all(checks))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("Dataset quality report generated.")
    print(f"ready_for_training: {report['ready_for_training']}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
