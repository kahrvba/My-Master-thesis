#!/usr/bin/env python3
"""
Feature extraction pipeline for adaptive sorting selection.

Implements v1 (cheap) features by default and optional v2 features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


def read_split(split_csv: Path) -> list[dict]:
    with split_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def resolve_sample_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(p)
        candidates.append(Path.cwd() / p)
        candidates.append(project_root / p)

        # Handle entries like "My-Master-thesis/data/..." when already inside project root
        if p.parts and p.parts[0] == "My-Master-thesis":
            stripped = Path(*p.parts[1:])
            candidates.append(stripped)
            candidates.append(Path.cwd() / stripped)
            candidates.append(project_root / stripped)

    # Deduplicate while preserving order
    seen = set()
    uniq_candidates: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            uniq_candidates.append(c)

    for c in uniq_candidates:
        if c.exists():
            return c

    tried = "\n  - ".join(str(c) for c in uniq_candidates)
    raise FileNotFoundError(
        f"Sample file not found for path '{raw_path}'. Tried:\n  - {tried}"
    )


def load_values(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".txt":
        text = path.read_text().strip()
        if not text:
            return np.array([], dtype=np.float64)
        arr = np.fromstring(text, sep=" ")
    elif suffix == ".csv":
        df = pd.read_csv(path)
        arr = df.to_numpy().reshape(-1)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return np.asarray(arr).reshape(-1)


def coerce_numeric_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u", "f"):
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
    arr = arr.astype(np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    return arr


def monotonic_run_stats(values: np.ndarray) -> tuple[int, int]:
    n = values.size
    if n == 0:
        return 0, 0
    if n == 1:
        return 1, 1

    diffs = np.diff(values.astype(np.float64, copy=False))
    runs = 1
    longest = 1
    current_len = 1
    direction = 0

    for d in diffs:
        if d == 0:
            current_len += 1
            continue
        new_dir = 1 if d > 0 else -1
        if direction == 0 or new_dir == direction:
            direction = new_dir
            current_len += 1
        else:
            longest = max(longest, current_len)
            runs += 1
            direction = new_dir
            current_len = 2
    longest = max(longest, current_len)
    return runs, longest


def inversion_count_merge(values: np.ndarray) -> int:
    arr = values.astype(np.float64, copy=False).tolist()

    def _sort_count(a: list[float]) -> tuple[list[float], int]:
        n = len(a)
        if n <= 1:
            return a, 0
        mid = n // 2
        left, inv_l = _sort_count(a[:mid])
        right, inv_r = _sort_count(a[mid:])
        merged: list[float] = []
        i = j = 0
        inv = inv_l + inv_r
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv += len(left) - i
                j += 1
        if i < len(left):
            merged.extend(left[i:])
        if j < len(right):
            merged.extend(right[j:])
        return merged, inv

    _, inv = _sort_count(arr)
    return int(inv)


def sample_rng(sample_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(local_seed)


def entropy_ratio(values: np.ndarray, bins: int = 32) -> float:
    if values.size <= 1:
        return 0.0
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < EPS:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p /= p.sum()
    ent = -np.sum(p * np.log2(p))
    max_ent = np.log2(float(bins))
    if max_ent < EPS:
        return 0.0
    return float(ent / max_ent)


def skewness(values: np.ndarray) -> float:
    if values.size <= 2:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    if std < EPS:
        return 0.0
    z = (values - mean) / (std + EPS)
    return float(np.mean(z**3))


def kurtosis_excess(values: np.ndarray) -> float:
    if values.size <= 3:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    if std < EPS:
        return 0.0
    z = (values - mean) / (std + EPS)
    return float(np.mean(z**4) - 3.0)


def signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def v1_features(values: np.ndarray, n_max_train: float) -> dict:
    n = int(values.size)
    if n == 0:
        return {
            "length_norm": 0.0,
            "adj_sorted_ratio": 1.0,
            "duplicate_ratio": 0.0,
            "dispersion_ratio": 0.0,
            "runs_ratio": 0.0,
        }

    length_norm = n / (n_max_train + EPS)
    vmin = float(values.min())
    vmax = float(values.max())
    value_range = vmax - vmin
    if n > 1:
        diffs = np.diff(values)
        adj_sorted_ratio = float(np.mean(diffs >= 0))
    else:
        adj_sorted_ratio = 1.0

    uniq = int(np.unique(values).size)
    duplicate_ratio = 1.0 - (uniq / n)

    std = float(values.std())
    dispersion_ratio = std / (value_range + EPS)
    runs_count, _ = monotonic_run_stats(values)
    runs_ratio = runs_count / n

    return {
        "length_norm": float(np.clip(length_norm, 0.0, 1.0)),
        "adj_sorted_ratio": float(np.clip(adj_sorted_ratio, 0.0, 1.0)),
        "duplicate_ratio": float(np.clip(duplicate_ratio, 0.0, 1.0)),
        "dispersion_ratio": float(np.clip(dispersion_ratio, 0.0, 1.0)),
        "runs_ratio": float(np.clip(runs_ratio, 0.0, 1.0)),
    }


def inversion_ratio(values: np.ndarray, sample_id: str, seed: int) -> float:
    n = int(values.size)
    if n <= 1:
        return 0.0
    if n <= 10_000:
        inv = inversion_count_merge(values)
        denom = (n * (n - 1)) / 2.0
        return float(inv / (denom + EPS))

    m = int(min(2000, n))
    rng = sample_rng(sample_id, seed)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    sampled = values[idx]
    inv = inversion_count_merge(sampled)
    denom = (m * (m - 1)) / 2.0
    return float(inv / (denom + EPS))


def v2_features(values: np.ndarray, sample_id: str, seed: int) -> dict:
    n = int(values.size)
    if n == 0:
        return {
            "inversion_ratio": 0.0,
            "entropy_ratio": 0.0,
            "skewness_t": 0.0,
            "kurtosis_excess_t": 0.0,
            "longest_run_ratio": 0.0,
            "iqr_norm": 0.0,
            "mad_norm": 0.0,
            "top1_freq_ratio": 0.0,
            "top5_freq_ratio": 0.0,
            "outlier_ratio": 0.0,
            "mean_abs_diff_norm": 0.0,
        }

    vmin = float(values.min())
    vmax = float(values.max())
    value_range = vmax - vmin

    runs_count, longest_run = monotonic_run_stats(values)
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    iqr_norm = (q75 - q25) / (value_range + EPS)

    med = float(np.median(values))
    mad_norm = float(np.median(np.abs(values - med)) / (value_range + EPS))

    uniq, counts = np.unique(values, return_counts=True)
    _ = uniq  # silence linters about unused variable
    counts = np.sort(counts)[::-1]
    top1_freq_ratio = float(counts[0] / n)
    top5_freq_ratio = float(counts[:5].sum() / n)

    std = float(values.std())
    if std < EPS:
        outlier_ratio = 0.0
    else:
        z = np.abs((values - float(values.mean())) / (std + EPS))
        outlier_ratio = float(np.mean(z > 3.0))

    if n > 1:
        mean_abs_diff_norm = float(np.mean(np.abs(np.diff(values))) / (value_range + EPS))
    else:
        mean_abs_diff_norm = 0.0

    skew = float(skewness(values))
    kurt = float(kurtosis_excess(values))

    return {
        "inversion_ratio": float(np.clip(inversion_ratio(values, sample_id, seed), 0.0, 1.0)),
        "entropy_ratio": float(np.clip(entropy_ratio(values), 0.0, 1.0)),
        "skewness_t": signed_log1p(skew),
        "kurtosis_excess_t": signed_log1p(kurt),
        "longest_run_ratio": float(np.clip(longest_run / n, 0.0, 1.0)),
        "iqr_norm": float(np.clip(iqr_norm, 0.0, 1.0)),
        "mad_norm": float(np.clip(mad_norm, 0.0, 1.0)),
        "top1_freq_ratio": float(np.clip(top1_freq_ratio, 0.0, 1.0)),
        "top5_freq_ratio": float(np.clip(top5_freq_ratio, 0.0, 1.0)),
        "outlier_ratio": float(np.clip(outlier_ratio, 0.0, 1.0)),
        "mean_abs_diff_norm": float(np.clip(mean_abs_diff_norm, 0.0, 1.0)),
    }


def compute_train_constants(
    train_rows: Iterable[dict], project_root: Path
) -> float:
    n_max = 0.0
    for row in train_rows:
        values = coerce_numeric_finite(
            load_values(resolve_sample_path(row["path"], project_root))
        )
        n = float(values.size)
        n_max = max(n_max, n)
    return max(n_max, 1.0)


def split_features(
    rows: list[dict],
    feature_set: str,
    n_max_train: float,
    seed: int,
    project_root: Path,
) -> pd.DataFrame:
    out_rows: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        values = coerce_numeric_finite(
            load_values(resolve_sample_path(row["path"], project_root))
        )
        feats = v1_features(values, n_max_train=n_max_train)

        rec = {"sample_id": sample_id, **feats}
        if feature_set == "v2":
            rec.update(v2_features(values, sample_id=sample_id, seed=seed))
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def write_parquet_strict(df: pd.DataFrame, base_path_no_ext: Path) -> dict:
    base_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = base_path_no_ext.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    return {
        "parquet_path": str(parquet_path.as_posix()),
    }


def validate_df(df: pd.DataFrame, feature_set: str) -> list[str]:
    errors: list[str] = []
    if df["sample_id"].duplicated().any():
        errors.append("Duplicate sample_id rows found.")

    feat_cols = [
        "length_norm",
        "adj_sorted_ratio",
        "duplicate_ratio",
        "dispersion_ratio",
        "runs_ratio",
    ]
    if feature_set == "v2":
        feat_cols.extend(
            [
                "inversion_ratio",
                "entropy_ratio",
                "skewness_t",
                "kurtosis_excess_t",
                "longest_run_ratio",
                "iqr_norm",
                "mad_norm",
                "top1_freq_ratio",
                "top5_freq_ratio",
                "outlier_ratio",
                "mean_abs_diff_norm",
            ]
        )

    numeric = df[feat_cols]
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append("NaN or inf values found in feature columns.")

    bounded_cols = [
        "length_norm",
        "adj_sorted_ratio",
        "duplicate_ratio",
        "dispersion_ratio",
        "runs_ratio",
    ]
    if feature_set == "v2":
        bounded_cols.extend(
            [
                "inversion_ratio",
                "entropy_ratio",
                "longest_run_ratio",
                "iqr_norm",
                "mad_norm",
                "top1_freq_ratio",
                "top5_freq_ratio",
                "outlier_ratio",
                "mean_abs_diff_norm",
            ]
        )
    for col in bounded_cols:
        bad = (df[col] < -1e-9) | (df[col] > 1.0 + 1e-9)
        if bad.any():
            errors.append(f"{col} has out-of-range values.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from dataset splits.")
    parser.add_argument("--splits-dir", default="My-Master-thesis/data/synthetic/splits")
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v1")
    parser.add_argument("--output-dir", default="My-Master-thesis/data/features")
    parser.add_argument("--artifacts-dir", default="My-Master-thesis/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_split(splits_dir / "train_ids.csv")
    val_rows = read_split(splits_dir / "val_ids.csv")
    test_rows = read_split(splits_dir / "test_ids.csv")

    n_max_train = compute_train_constants(train_rows, project_root=project_root)

    train_df = split_features(
        train_rows,
        args.feature_set,
        n_max_train=n_max_train,
        seed=args.seed,
        project_root=project_root,
    )
    val_df = split_features(
        val_rows,
        args.feature_set,
        n_max_train=n_max_train,
        seed=args.seed,
        project_root=project_root,
    )
    test_df = split_features(
        test_rows,
        args.feature_set,
        n_max_train=n_max_train,
        seed=args.seed,
        project_root=project_root,
    )

    errors: list[str] = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        e = validate_df(df, feature_set=args.feature_set)
        errors.extend([f"{name}: {msg}" for msg in e])

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    train_out = write_parquet_strict(train_df, output_dir / "train_features")
    val_out = write_parquet_strict(val_df, output_dir / "val_features")
    test_out = write_parquet_strict(test_df, output_dir / "test_features")

    config = {
        "feature_set": args.feature_set,
        "seed": args.seed,
        "constants": {
            "n_max_train": n_max_train,
        },
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "outputs": {
            "train": train_out,
            "val": val_out,
            "test": test_out,
        },
    }
    (artifacts_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    print("Feature extraction complete.")
    print(f"Feature set: {args.feature_set}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"n_max_train: {n_max_train}")
    print("Validation checks passed.")


if __name__ == "__main__":
    main()
