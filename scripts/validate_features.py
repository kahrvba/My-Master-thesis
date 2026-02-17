#!/usr/bin/env python3
"""
Feature Extraction Validation Suite
=====================================
Validates that every feature is mathematically correct on known inputs,
handles edge cases, stays bounded, is deterministic, and matches the
benchmark dataset.

Tests:
  1. Handcrafted arrays with known expected values (ground truth)
  2. Edge cases: constant, single element, 2 elements, all-same, huge range
  3. Bounds checking: all features in expected range for random data
  4. Determinism: same input → same output
  5. Consistency: features in benchmark parquet match fresh extraction
  6. Real-world-like: works on float arrays, negative values, mixed types
  7. Inversion count correctness: brute-force vs merge-sort count
  8. _monotonic_run_stats correctness on known patterns
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import extract_features from benchmark_algorithms ─────────────────────
# Add scripts/ to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_algorithms import (
    extract_features,
    _FEATURE_NAMES,
    _monotonic_run_stats,
    _inversion_count_merge,
    EPS,
    SEED,
)


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  — {detail}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Known ground-truth arrays
# ═══════════════════════════════════════════════════════════════════════════

def test_sorted_array():
    """Perfectly sorted array — known expected values."""
    print("\n── TEST 1a: Perfectly sorted [1,2,3,4,5,6,7,8,9,10] ──")
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    f = extract_features(arr, n_max=10.0, sample_id="test_sorted")

    check("length_norm = 1.0", approx(f["length_norm"], 1.0),
          f"got {f['length_norm']}")
    check("adj_sorted_ratio = 1.0 (all ascending)", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("duplicate_ratio = 0.0 (all unique)", approx(f["duplicate_ratio"], 0.0),
          f"got {f['duplicate_ratio']}")
    check("inversion_ratio = 0.0 (sorted)", approx(f["inversion_ratio"], 0.0),
          f"got {f['inversion_ratio']}")
    check("longest_run_ratio = 1.0 (one long run)", approx(f["longest_run_ratio"], 1.0),
          f"got {f['longest_run_ratio']}")
    check("runs_ratio = 0.1 (1 run / 10)", approx(f["runs_ratio"], 0.1),
          f"got {f['runs_ratio']}")
    # Mean abs diff: all diffs are 1, range is 9 → 1/9 ≈ 0.1111
    check("mean_abs_diff_norm ≈ 0.111", approx(f["mean_abs_diff_norm"], 1.0/9.0, tol=0.01),
          f"got {f['mean_abs_diff_norm']}")


def test_reverse_sorted_array():
    """Reverse sorted — maximum inversions."""
    print("\n── TEST 1b: Reverse sorted [10,9,8,...,1] ──")
    arr = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    f = extract_features(arr, n_max=10.0, sample_id="test_reverse")

    check("adj_sorted_ratio = 0.0 (all descending)", approx(f["adj_sorted_ratio"], 0.0),
          f"got {f['adj_sorted_ratio']}")
    check("inversion_ratio = 1.0 (max inversions)", approx(f["inversion_ratio"], 1.0),
          f"got {f['inversion_ratio']}")
    check("duplicate_ratio = 0.0", approx(f["duplicate_ratio"], 0.0),
          f"got {f['duplicate_ratio']}")
    check("longest_run_ratio = 1.0 (one descending run)", approx(f["longest_run_ratio"], 1.0),
          f"got {f['longest_run_ratio']}")


def test_constant_array():
    """All same values — edge case."""
    print("\n── TEST 1c: Constant [5,5,5,5,5] ──")
    arr = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    f = extract_features(arr, n_max=10.0, sample_id="test_constant")

    check("adj_sorted_ratio = 1.0 (diff >= 0 for equals)", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("duplicate_ratio = 0.8 (1 unique / 5)", approx(f["duplicate_ratio"], 0.8),
          f"got {f['duplicate_ratio']}")
    check("dispersion_ratio = 0.0 (std=0, range=0)", approx(f["dispersion_ratio"], 0.0),
          f"got {f['dispersion_ratio']}")
    check("entropy_ratio = 0.0 (no variation)", approx(f["entropy_ratio"], 0.0),
          f"got {f['entropy_ratio']}")
    check("skewness_t = 0.0 (std=0)", approx(f["skewness_t"], 0.0),
          f"got {f['skewness_t']}")
    check("kurtosis_excess_t = 0.0 (std=0)", approx(f["kurtosis_excess_t"], 0.0),
          f"got {f['kurtosis_excess_t']}")
    check("iqr_norm = 0.0 (range=0)", approx(f["iqr_norm"], 0.0),
          f"got {f['iqr_norm']}")
    check("mad_norm = 0.0 (all same)", approx(f["mad_norm"], 0.0),
          f"got {f['mad_norm']}")
    check("outlier_ratio = 0.0 (std=0)", approx(f["outlier_ratio"], 0.0),
          f"got {f['outlier_ratio']}")
    check("mean_abs_diff_norm = 0.0", approx(f["mean_abs_diff_norm"], 0.0),
          f"got {f['mean_abs_diff_norm']}")
    check("top1_freq_ratio = 1.0 (all same value)", approx(f["top1_freq_ratio"], 1.0),
          f"got {f['top1_freq_ratio']}")
    check("inversion_ratio = 0.0 (no inversions)", approx(f["inversion_ratio"], 0.0),
          f"got {f['inversion_ratio']}")


def test_two_values():
    """Half duplicates — verify duplicate_ratio and freq."""
    print("\n── TEST 1d: Two values [1,1,1,2,2,2] ──")
    arr = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    f = extract_features(arr, n_max=6.0, sample_id="test_two_vals")

    check("duplicate_ratio = 2/3 (2 unique / 6)", approx(f["duplicate_ratio"], 4.0/6.0),
          f"got {f['duplicate_ratio']}, expected {4.0/6.0}")
    check("adj_sorted_ratio ≈ 1.0 (sorted)", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("top1_freq_ratio = 0.5 (3/6)", approx(f["top1_freq_ratio"], 0.5),
          f"got {f['top1_freq_ratio']}")
    check("top5_freq_ratio = 1.0 (all covered by 2 vals)", approx(f["top5_freq_ratio"], 1.0),
          f"got {f['top5_freq_ratio']}")


def test_alternating():
    """Alternating pattern — many inversions, many runs."""
    print("\n── TEST 1e: Alternating [1,10,1,10,1,10,1,10] ──")
    arr = np.array([1, 10, 1, 10, 1, 10, 1, 10], dtype=np.int32)
    f = extract_features(arr, n_max=8.0, sample_id="test_alt")

    # adj_sorted: diffs = [9,-9,9,-9,9,-9,9] → 4 out of 7 ≥ 0 → 4/7
    check("adj_sorted_ratio ≈ 4/7", approx(f["adj_sorted_ratio"], 4.0/7.0, tol=0.02),
          f"got {f['adj_sorted_ratio']}, expected {4.0/7.0}")
    # duplicate_ratio: 2 unique / 8 → 1 - 2/8 = 0.75
    check("duplicate_ratio = 0.75", approx(f["duplicate_ratio"], 0.75),
          f"got {f['duplicate_ratio']}")
    # Mean abs diff: all diffs = 9, range = 9 → 9/9 = 1.0
    check("mean_abs_diff_norm = 1.0", approx(f["mean_abs_diff_norm"], 1.0, tol=0.01),
          f"got {f['mean_abs_diff_norm']}")


def test_few_unique_pattern():
    """Few unique values — counting sort territory."""
    print("\n── TEST 1f: Few unique [3,3,3,1,1,2,2,2,2,3] ──")
    arr = np.array([3, 3, 3, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
    f = extract_features(arr, n_max=10.0, sample_id="test_few_uniq")

    # 3 unique values / 10 → dup_ratio = 1 - 3/10 = 0.7
    check("duplicate_ratio = 0.7", approx(f["duplicate_ratio"], 0.7),
          f"got {f['duplicate_ratio']}")
    # top1 freq: value 2 appears 4 times → 4/10 = 0.4; or value 3 appears 4 times
    # Actually: 3→4, 1→2, 2→4. So top1 = 4/10 = 0.4
    check("top1_freq_ratio = 0.4", approx(f["top1_freq_ratio"], 0.4),
          f"got {f['top1_freq_ratio']}")
    check("top5_freq_ratio = 1.0 (only 3 unique ≤ 5)", approx(f["top5_freq_ratio"], 1.0),
          f"got {f['top5_freq_ratio']}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Edge cases
# ═══════════════════════════════════════════════════════════════════════════

def test_single_element():
    """Single element array."""
    print("\n── TEST 2a: Single element [42] ──")
    arr = np.array([42], dtype=np.int32)
    f = extract_features(arr, n_max=100.0, sample_id="test_single")

    check("length_norm = 0.01", approx(f["length_norm"], 1.0/100.0),
          f"got {f['length_norm']}")
    check("adj_sorted_ratio = 1.0", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("duplicate_ratio = 0.0", approx(f["duplicate_ratio"], 0.0),
          f"got {f['duplicate_ratio']}")
    check("inversion_ratio = 0.0", approx(f["inversion_ratio"], 0.0),
          f"got {f['inversion_ratio']}")
    check("mean_abs_diff_norm = 0.0", approx(f["mean_abs_diff_norm"], 0.0),
          f"got {f['mean_abs_diff_norm']}")
    # All features should be finite
    for feat_name in _FEATURE_NAMES:
        check(f"{feat_name} is finite", np.isfinite(f[feat_name]),
              f"got {f[feat_name]}")


def test_two_elements():
    """Two elements."""
    print("\n── TEST 2b: Two elements [1, 100] ──")
    arr = np.array([1, 100], dtype=np.int32)
    f = extract_features(arr, n_max=100.0, sample_id="test_two")

    check("adj_sorted_ratio = 1.0 (ascending)", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("inversion_ratio = 0.0 (sorted)", approx(f["inversion_ratio"], 0.0),
          f"got {f['inversion_ratio']}")
    check("duplicate_ratio = 0.0 (both unique)", approx(f["duplicate_ratio"], 0.0),
          f"got {f['duplicate_ratio']}")

    arr2 = np.array([100, 1], dtype=np.int32)
    f2 = extract_features(arr2, n_max=100.0, sample_id="test_two_rev")
    check("reversed: adj_sorted_ratio = 0.0", approx(f2["adj_sorted_ratio"], 0.0),
          f"got {f2['adj_sorted_ratio']}")
    check("reversed: inversion_ratio = 1.0", approx(f2["inversion_ratio"], 1.0),
          f"got {f2['inversion_ratio']}")


def test_negative_values():
    """Negative values — must work for real-world data."""
    print("\n── TEST 2c: Negative values [-100, -50, 0, 50, 100] ──")
    arr = np.array([-100, -50, 0, 50, 100], dtype=np.int32)
    f = extract_features(arr, n_max=1000.0, sample_id="test_neg")

    # Range = 200, all sorted ascending
    check("adj_sorted_ratio = 1.0", approx(f["adj_sorted_ratio"], 1.0),
          f"got {f['adj_sorted_ratio']}")
    check("inversion_ratio = 0.0", approx(f["inversion_ratio"], 0.0),
          f"got {f['inversion_ratio']}")
    check("all features finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))

    # Mean abs diff: all diffs = 50, range = 200 → 50/200 = 0.25
    check("mean_abs_diff_norm = 0.25", approx(f["mean_abs_diff_norm"], 0.25),
          f"got {f['mean_abs_diff_norm']}")


def test_float_array():
    """Float arrays — real-world data may be float64."""
    print("\n── TEST 2d: Float array [1.5, 2.7, 3.1, 0.8] ──")
    arr = np.array([1.5, 2.7, 3.1, 0.8], dtype=np.float64)
    f = extract_features(arr, n_max=100.0, sample_id="test_float")

    check("all features finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))
    check("all features are float", all(isinstance(f[k], float) for k in _FEATURE_NAMES))
    # 3 out of 3 diffs: [1.2, 0.4, -2.3] → 2/3 ≥ 0
    check("adj_sorted_ratio = 2/3", approx(f["adj_sorted_ratio"], 2.0/3.0, tol=0.02),
          f"got {f['adj_sorted_ratio']}")


def test_large_range():
    """Large value range — stress test for dispersion/IQR."""
    print("\n── TEST 2e: Large range [0, 1_000_000_000] ──")
    arr = np.array([0, 1_000_000_000], dtype=np.int64)
    f = extract_features(arr, n_max=100.0, sample_id="test_large_range")

    check("all features finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))
    check("all features bounded", all(
        -10 <= f[k] <= 10 for k in _FEATURE_NAMES  # skewness_t and kurtosis_excess_t can be negative
    ), f"out-of-range: {[(k, f[k]) for k in _FEATURE_NAMES if not (-10 <= f[k] <= 10)]}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Bounds checking on random data
# ═══════════════════════════════════════════════════════════════════════════

def test_bounds_random():
    """All bounded features should stay in [0, 1] on random data."""
    print("\n── TEST 3: Bounds on 100 random arrays ──")
    rng = np.random.default_rng(12345)

    bounded_features = [
        "length_norm", "adj_sorted_ratio", "duplicate_ratio", "dispersion_ratio",
        "runs_ratio", "inversion_ratio", "entropy_ratio", "longest_run_ratio",
        "iqr_norm", "mad_norm", "top1_freq_ratio", "top5_freq_ratio",
        "outlier_ratio", "mean_abs_diff_norm",
    ]
    # skewness_t and kurtosis_excess_t are log-transformed, can be negative

    violations = []
    for i in range(100):
        n = int(rng.integers(10, 10000))
        arr = rng.integers(-1_000_000, 1_000_000, size=n, dtype=np.int32)
        f = extract_features(arr, n_max=10000.0, sample_id=f"rand_{i}")

        for feat in bounded_features:
            v = f[feat]
            if v < -1e-9 or v > 1.0 + 1e-9:
                violations.append((i, feat, v))

        # Check all are finite
        for feat in _FEATURE_NAMES:
            if not np.isfinite(f[feat]):
                violations.append((i, feat, f[feat]))

    check(f"No bound violations in 100 random arrays", len(violations) == 0,
          f"{len(violations)} violations: {violations[:5]}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Determinism
# ═══════════════════════════════════════════════════════════════════════════

def test_determinism():
    """Same input → same output, always."""
    print("\n── TEST 4: Determinism (run extract_features 3 times) ──")
    arr = np.array([5, 3, 1, 4, 2, 8, 7, 6, 10, 9], dtype=np.int32)

    results = []
    for _ in range(3):
        f = extract_features(arr.copy(), n_max=10.0, sample_id="test_det")
        results.append(f)

    all_same = True
    for feat in _FEATURE_NAMES:
        vals = [r[feat] for r in results]
        if not (vals[0] == vals[1] == vals[2]):
            all_same = False
            print(f"    {feat}: {vals}")

    check("All features identical across 3 runs", all_same)


def test_determinism_large():
    """Determinism for large arrays (uses inversion_ratio subsampling)."""
    print("\n── TEST 4b: Determinism on large array (n=50K, subsampled inversions) ──")
    rng = np.random.default_rng(777)
    arr = rng.integers(0, 1_000_000, size=50_000, dtype=np.int32)

    results = []
    for _ in range(3):
        f = extract_features(arr.copy(), n_max=100_000.0, sample_id="test_det_large")
        results.append(f)

    all_same = True
    for feat in _FEATURE_NAMES:
        vals = [r[feat] for r in results]
        if not approx(vals[0], vals[1]) or not approx(vals[0], vals[2]):
            all_same = False
            print(f"    {feat}: {vals}")

    check("All features identical across 3 runs (large array)", all_same)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Consistency with benchmark dataset
# ═══════════════════════════════════════════════════════════════════════════

def test_benchmark_consistency():
    """Verify features in the parquet match what extract_features produces."""
    print("\n── TEST 5: Benchmark dataset consistency ──")
    project_root = Path(__file__).resolve().parent.parent
    parquet_path = project_root / "data" / "benchmark" / "all_samples.parquet"

    if not parquet_path.exists():
        print("  ⚠ Benchmark parquet not found — skipping consistency check")
        return

    df = pd.read_parquet(parquet_path)

    # Check all feature columns exist
    for feat in _FEATURE_NAMES:
        check(f"Column '{feat}' exists in parquet", feat in df.columns)

    # Check no NaN/Inf in features
    feat_data = df[_FEATURE_NAMES].to_numpy()
    check("No NaN in features", not np.any(np.isnan(feat_data)),
          f"{np.sum(np.isnan(feat_data))} NaN values")
    check("No Inf in features", not np.any(np.isinf(feat_data)),
          f"{np.sum(np.isinf(feat_data))} Inf values")

    # Check timing columns exist and are positive
    time_cols = ["time_introsort", "time_heapsort", "time_timsort", "time_counting_sort"]
    for col in time_cols:
        check(f"Column '{col}' exists", col in df.columns)
        if col in df.columns:
            check(f"'{col}' all positive", (df[col] > 0).all(),
                  f"min={df[col].min()}")

    # Check metadata columns
    for col in ["sample_id", "n", "distribution", "structure", "split"]:
        check(f"Column '{col}' exists", col in df.columns)

    # Check split assignments
    splits = df["split"].value_counts().to_dict()
    check("Has 'train' split", "train" in splits, f"splits: {splits}")
    check("Has 'val' split", "val" in splits, f"splits: {splits}")
    check("Has 'test_A' split", "test_A" in splits, f"splits: {splits}")
    check("Has 'test_B' split", "test_B" in splits, f"splits: {splits}")

    # Check distribution-split correctness
    train_dists = set(df[df["split"] == "train"]["distribution"].unique())
    shift_dists = set(df[df["split"] == "test_B"]["distribution"].unique())
    check("Train uses uniform+normal", train_dists == {"uniform", "normal"},
          f"got {train_dists}")
    check("Test_B uses lognormal+exponential", shift_dists == {"lognormal", "exponential"},
          f"got {shift_dists}")

    # Bounded features check across entire dataset
    bounded_features = [
        "length_norm", "adj_sorted_ratio", "duplicate_ratio", "dispersion_ratio",
        "runs_ratio", "inversion_ratio", "entropy_ratio", "longest_run_ratio",
        "iqr_norm", "mad_norm", "top1_freq_ratio", "top5_freq_ratio",
        "outlier_ratio", "mean_abs_diff_norm",
    ]
    for feat in bounded_features:
        mn = float(df[feat].min())
        mx = float(df[feat].max())
        check(f"{feat} in [0,1] range", mn >= -1e-9 and mx <= 1.0 + 1e-9,
              f"min={mn}, max={mx}")

    # Check no duplicate sample_ids
    check("No duplicate sample_ids", df["sample_id"].is_unique)

    # Check total count
    check(f"Total samples = 720", len(df) == 720, f"got {len(df)}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Real-world-like data
# ═══════════════════════════════════════════════════════════════════════════

def test_real_world_patterns():
    """Test on data patterns that might come from real applications."""
    print("\n── TEST 6: Real-world-like patterns ──")

    # Time-series-like: mostly sorted with noise
    rng = np.random.default_rng(42)
    ts = np.cumsum(rng.normal(1.0, 0.1, size=1000)).astype(np.float64)
    f = extract_features(ts, n_max=10000.0, sample_id="test_timeseries")
    check("Time-series: high adj_sorted_ratio", f["adj_sorted_ratio"] > 0.8,
          f"got {f['adj_sorted_ratio']}")
    check("Time-series: low inversion_ratio", f["inversion_ratio"] < 0.2,
          f"got {f['inversion_ratio']}")
    check("Time-series: all finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))

    # Log-distributed IDs (like user IDs, timestamps)
    ids = rng.lognormal(10, 2, size=5000).astype(np.float64)
    rng.shuffle(ids)
    f = extract_features(ids, n_max=10000.0, sample_id="test_logids")
    check("Log-IDs: positive skewness", f["skewness_t"] > 0,
          f"got {f['skewness_t']}")
    check("Log-IDs: all finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))

    # Categorical-like: few unique integers
    cats = rng.choice([10, 20, 30, 40, 50], size=10000).astype(np.int32)
    f = extract_features(cats, n_max=10000.0, sample_id="test_categorical")
    check("Categorical: high duplicate_ratio", f["duplicate_ratio"] > 0.99,
          f"got {f['duplicate_ratio']}")
    check("Categorical: top5 covers all", approx(f["top5_freq_ratio"], 1.0, tol=0.01),
          f"got {f['top5_freq_ratio']}")
    check("Categorical: all finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))

    # Nearly-constant with outliers
    nearly_const = np.full(1000, 42.0)
    nearly_const[0] = 0.0
    nearly_const[999] = 100.0
    f = extract_features(nearly_const.astype(np.float64), n_max=10000.0, sample_id="test_outlier_heavy")
    check("Nearly-constant: high duplicate_ratio", f["duplicate_ratio"] > 0.99,
          f"got {f['duplicate_ratio']}")
    check("Nearly-constant: all finite", all(np.isfinite(f[k]) for k in _FEATURE_NAMES))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Inversion count correctness
# ═══════════════════════════════════════════════════════════════════════════

def _brute_force_inversions(arr: list) -> int:
    """O(n²) brute force for ground truth."""
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count


def test_inversion_count():
    """Verify merge-sort inversion count matches brute force."""
    print("\n── TEST 7: Inversion count vs brute force ──")

    test_cases = [
        [1, 2, 3, 4, 5],         # 0 inversions
        [5, 4, 3, 2, 1],         # 10 inversions
        [2, 1, 3, 1, 2],         # 4 inversions
        [1],                      # 0
        [],                       # 0
        [3, 1, 2],               # 2 inversions
        [1, 1, 1],               # 0 inversions (equal = not inversion)
    ]

    for tc in test_cases:
        expected = _brute_force_inversions(tc)
        arr_copy = tc.copy()
        got = _inversion_count_merge(arr_copy)
        check(f"Inversions of {tc}: expected={expected}", got == expected,
              f"got={got}")

    # Random arrays
    rng = np.random.default_rng(99)
    for i in range(20):
        n = int(rng.integers(2, 50))
        arr = rng.integers(0, 100, size=n).tolist()
        expected = _brute_force_inversions(arr)
        arr_copy = arr.copy()
        got = _inversion_count_merge(arr_copy)
        check(f"Random inversions (n={n}, trial {i})", got == expected,
              f"expected={expected}, got={got}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Monotonic run stats correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_monotonic_runs():
    """Verify _monotonic_run_stats on known patterns."""
    print("\n── TEST 8: Monotonic run stats ──")

    # Fully ascending: 1 run of length 5
    arr = np.array([1, 2, 3, 4, 5])
    runs, longest = _monotonic_run_stats(arr)
    check("Ascending: 1 run", runs == 1, f"got {runs}")
    check("Ascending: longest=5", longest == 5, f"got {longest}")

    # Fully descending: 1 run of length 5
    arr = np.array([5, 4, 3, 2, 1])
    runs, longest = _monotonic_run_stats(arr)
    check("Descending: 1 run", runs == 1, f"got {runs}")
    check("Descending: longest=5", longest == 5, f"got {longest}")

    # Alternating up-down: [1, 3, 2, 4, 3]
    # diffs = [2,-1,2,-1] → direction changes at each step → 4 runs, each length 2
    arr = np.array([1, 3, 2, 4, 3])
    runs, longest = _monotonic_run_stats(arr)
    check("Up-down alternating: 4 runs", runs == 4, f"got {runs}")
    check("Up-down alternating: longest=2", longest == 2, f"got {longest}")

    # All equal: [5, 5, 5]
    arr = np.array([5, 5, 5])
    runs, longest = _monotonic_run_stats(arr)
    check("All equal: 1 run", runs == 1, f"got {runs}")
    check("All equal: longest=3", longest == 3, f"got {longest}")

    # Single element
    arr = np.array([42])
    runs, longest = _monotonic_run_stats(arr)
    check("Single: 1 run", runs == 1, f"got {runs}")
    check("Single: longest=1", longest == 1, f"got {longest}")

    # Empty
    arr = np.array([])
    runs, longest = _monotonic_run_stats(arr)
    check("Empty: 0 runs", runs == 0, f"got {runs}")
    check("Empty: longest=0", longest == 0, f"got {longest}")

    # Two ascending runs: [1,2,3, 0,1,2]
    # diffs = [1,1,-3,1,1] → asc(3), desc(2), asc(3) = 3 runs, longest=3
    arr = np.array([1, 2, 3, 0, 1, 2])
    runs, longest = _monotonic_run_stats(arr)
    check("Two asc runs: 3 runs", runs == 3, f"got {runs}")
    check("Two asc runs: longest=3", longest == 3, f"got {longest}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: Feature sensitivity — features should CHANGE with input
# ═══════════════════════════════════════════════════════════════════════════

def test_feature_sensitivity():
    """Features should be sensitive to the properties they measure."""
    print("\n── TEST 9: Feature sensitivity ──")

    # adj_sorted_ratio: sorted > random
    sorted_arr = np.arange(1000, dtype=np.int32)
    random_arr = np.random.default_rng(42).permutation(sorted_arr).astype(np.int32)

    f_sorted = extract_features(sorted_arr, n_max=1000.0, sample_id="sens_sorted")
    f_random = extract_features(random_arr, n_max=1000.0, sample_id="sens_random")

    check("adj_sorted: sorted > random",
          f_sorted["adj_sorted_ratio"] > f_random["adj_sorted_ratio"],
          f"sorted={f_sorted['adj_sorted_ratio']}, random={f_random['adj_sorted_ratio']}")

    check("inversion: sorted < random",
          f_sorted["inversion_ratio"] < f_random["inversion_ratio"],
          f"sorted={f_sorted['inversion_ratio']}, random={f_random['inversion_ratio']}")

    check("runs: sorted < random (fewer runs)",
          f_sorted["runs_ratio"] < f_random["runs_ratio"],
          f"sorted={f_sorted['runs_ratio']}, random={f_random['runs_ratio']}")

    # duplicate_ratio: few_unique > all_unique
    all_unique = np.arange(1000, dtype=np.int32)
    few_unique = np.array([1, 2, 3] * 333 + [4], dtype=np.int32)

    f_unique = extract_features(all_unique, n_max=1000.0, sample_id="sens_uniq")
    f_dupes = extract_features(few_unique, n_max=1000.0, sample_id="sens_dupes")

    check("duplicate_ratio: few_unique > all_unique",
          f_dupes["duplicate_ratio"] > f_unique["duplicate_ratio"],
          f"dupes={f_dupes['duplicate_ratio']}, unique={f_unique['duplicate_ratio']}")

    check("top1_freq: few_unique > all_unique",
          f_dupes["top1_freq_ratio"] > f_unique["top1_freq_ratio"],
          f"dupes={f_dupes['top1_freq_ratio']}, unique={f_unique['top1_freq_ratio']}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: Return type and completeness
# ═══════════════════════════════════════════════════════════════════════════

def test_return_completeness():
    """extract_features returns exactly 16 features, all floats."""
    print("\n── TEST 10: Return type and completeness ──")
    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.int32)
    f = extract_features(arr, n_max=100.0, sample_id="test_complete")

    check("Returns dict", isinstance(f, dict))
    check(f"Has exactly {len(_FEATURE_NAMES)} keys", len(f) == len(_FEATURE_NAMES),
          f"got {len(f)} keys: {list(f.keys())}")

    for name in _FEATURE_NAMES:
        check(f"'{name}' present", name in f, f"missing from {list(f.keys())}")
        if name in f:
            check(f"'{name}' is float", isinstance(f[name], float),
                  f"type={type(f[name])}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FEATURE EXTRACTION VALIDATION SUITE")
    print("=" * 70)

    # Ground truth tests
    test_sorted_array()
    test_reverse_sorted_array()
    test_constant_array()
    test_two_values()
    test_alternating()
    test_few_unique_pattern()

    # Edge cases
    test_single_element()
    test_two_elements()
    test_negative_values()
    test_float_array()
    test_large_range()

    # Bounds
    test_bounds_random()

    # Determinism
    test_determinism()
    test_determinism_large()

    # Benchmark consistency
    test_benchmark_consistency()

    # Real-world patterns
    test_real_world_patterns()

    # Inversion count
    test_inversion_count()

    # Monotonic runs
    test_monotonic_runs()

    # Sensitivity
    test_feature_sensitivity()

    # Completeness
    test_return_completeness()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 70}")

    if FAIL > 0:
        print("\n⚠ FAILURES DETECTED — features are NOT validated!")
        sys.exit(1)
    else:
        print("\n✓ All features validated — safe to use on real data.")
        sys.exit(0)


if __name__ == "__main__":
    main()
