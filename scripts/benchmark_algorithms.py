#!/usr/bin/env python3
"""
Step 2: Full Timing Pipeline
==============================
Generates arrays at scale → extracts features → times 4 algorithms → saves dataset.

NO raw .npy files saved — reproducible via deterministic seeds.
Each row in the output = (sample_id, metadata, 16 features, 4 timing columns).

Split design (for bandit experiment):
  - uniform + normal → train (60%) / val (20%) / test_A (20%)     [XGBoost trained on these]
  - lognormal + exponential → test_B (100%)                        [bandit adapts to these]

Usage:
    source venv/bin/activate
    python scripts/benchmark_algorithms.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────

SEED = 42
EPS = 1e-12
VALUE_LOW = 0
VALUE_HIGH = 30_000_000
TIMING_REPEATS_SMALL = 7    # n <= 500K: more repeats for stability
TIMING_REPEATS_LARGE = 5    # n > 500K: fewer repeats (each is expensive)

SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]
DISTRIBUTIONS = ["uniform", "normal", "lognormal", "exponential"]
STRUCTURES = ["random", "nearly_sorted", "reverse_sorted", "few_unique", "sorted_runs", "sorted"]
REPEATS = 5  # per (size, distribution, structure) combination

# Total samples = 6 sizes × 4 dists × 6 structs × 5 repeats = 720

# Distributions for train vs bandit-shift
TRAIN_DISTS = {"uniform", "normal"}
SHIFT_DISTS = {"lognormal", "exponential"}


# ─── Array Generation (reuses logic from generate_synthetic_dataset.py) ───

def generate_base_array(
    rng: np.random.Generator, n: int, distribution: str
) -> np.ndarray:
    """Generate base values from a given distribution."""
    if distribution == "uniform":
        arr = rng.integers(VALUE_LOW, VALUE_HIGH, size=n, dtype=np.int64)
    elif distribution == "normal":
        mean = (VALUE_LOW + VALUE_HIGH) / 2.0
        std = (VALUE_HIGH - VALUE_LOW) / 6.0
        arr = rng.normal(loc=mean, scale=std, size=n)
        arr = np.clip(arr, VALUE_LOW, VALUE_HIGH - 1).astype(np.int64)
    elif distribution == "lognormal":
        arr = rng.lognormal(mean=2.5, sigma=1.0, size=n)
        arr = _scale_to_int_range(arr, VALUE_LOW, VALUE_HIGH)
    elif distribution == "exponential":
        arr = rng.exponential(scale=1.0, size=n)
        arr = _scale_to_int_range(arr, VALUE_LOW, VALUE_HIGH)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return arr.astype(np.int32, copy=False)


def _scale_to_int_range(arr: np.ndarray, low: int, high: int) -> np.ndarray:
    arr = arr.astype(np.float64, copy=False)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < EPS:
        return np.full_like(arr, low, dtype=np.int64)
    scaled = (arr - mn) / (mx - mn) * (high - low - 1) + low
    return scaled.astype(np.int64)


def apply_structure(
    rng: np.random.Generator, arr: np.ndarray, structure: str
) -> np.ndarray:
    """Apply a structural pattern to the array."""
    n = arr.size
    out = arr.copy()

    if structure == "random":
        rng.shuffle(out)
        return out

    if structure == "nearly_sorted":
        out.sort()
        swaps = max(1, int(0.02 * n))
        i_idx = rng.integers(0, n, size=swaps)
        j_idx = rng.integers(0, n, size=swaps)
        out[i_idx], out[j_idx] = out[j_idx].copy(), out[i_idx].copy()
        return out

    if structure == "reverse_sorted":
        out.sort()
        return out[::-1].copy()

    if structure == "few_unique":
        uniq = max(2, min(16, n // 100))
        unique_values = rng.choice(out, size=uniq, replace=False)
        return rng.choice(unique_values, size=n, replace=True).astype(np.int32)

    if structure == "sorted_runs":
        rng.shuffle(out)
        chunks: list[np.ndarray] = []
        i = 0
        while i < n:
            run_len = min(int(rng.integers(64, 512)), n - i)
            chunk = np.sort(out[i:i + run_len])
            if rng.random() < 0.5:
                chunk = chunk[::-1]
            chunks.append(chunk)
            i += run_len
        return np.concatenate(chunks).astype(np.int32, copy=False)

    if structure == "sorted":
        out.sort()
        return out

    raise ValueError(f"Unknown structure: {structure}")


# ─── Feature Extraction (reuses logic from extract_features.py) ──────────

def _monotonic_run_stats(values: np.ndarray) -> tuple[int, int]:
    """Count monotonic runs and longest run length. O(n)."""
    n = values.size
    if n <= 1:
        return max(n, 0), max(n, 0)

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


def _inversion_count_merge(arr: list[float]) -> int:
    """Merge-sort-based inversion count. O(n log n)."""
    n = len(arr)
    if n <= 1:
        return 0
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]
    inv_l = _inversion_count_merge(left)
    inv_r = _inversion_count_merge(right)

    # Inline merge to avoid function call overhead
    merged = []
    i = j = inv = 0
    inv += inv_l + inv_r
    ll = len(left)
    while i < ll and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv += ll - i
            j += 1
    if i < ll:
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])
    arr[:] = merged
    return inv


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def extract_features(values: np.ndarray, n_max: float, sample_id: str) -> dict:
    """
    Extract all 16 v2 features. Returns a flat dict of feature values.

    For inversion_ratio on large arrays (>10K), we subsample 2000 elements
    to keep cost manageable.
    """
    n = int(values.size)
    if n == 0:
        # Return zeros for empty arrays (shouldn't happen in practice)
        return {k: 0.0 for k in _FEATURE_NAMES}

    vals = values.astype(np.float64, copy=False)
    vmin = float(vals.min())
    vmax = float(vals.max())
    value_range = vmax - vmin
    std = float(vals.std())
    mean = float(vals.mean())
    med = float(np.median(vals))

    # ── v1 features ──
    length_norm = float(np.clip(n / (n_max + EPS), 0.0, 1.0))

    diffs = np.diff(vals)
    adj_sorted_ratio = float(np.mean(diffs >= 0)) if n > 1 else 1.0

    uniq_vals, uniq_counts = np.unique(vals, return_counts=True)
    n_unique = int(uniq_vals.size)
    duplicate_ratio = float(1.0 - n_unique / n)

    dispersion_ratio = float(np.clip(std / (value_range + EPS), 0.0, 1.0))

    runs_count, longest_run = _monotonic_run_stats(values)
    runs_ratio = float(np.clip(runs_count / n, 0.0, 1.0))

    # ── v2 features ──

    # Inversion ratio (expensive — subsample for large arrays)
    if n <= 10_000:
        inv = _inversion_count_merge(vals.tolist())
        denom = (n * (n - 1)) / 2.0
    else:
        m = 2000
        digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng_sub = np.random.default_rng(local_seed)
        idx = np.sort(rng_sub.choice(n, size=m, replace=False))
        sampled = vals[idx].tolist()
        inv = _inversion_count_merge(sampled)
        denom = (m * (m - 1)) / 2.0
    inversion_ratio = float(np.clip(inv / (denom + EPS), 0.0, 1.0))

    # Entropy ratio (32-bin histogram)
    if value_range < EPS:
        entropy_ratio = 0.0
    else:
        hist, _ = np.histogram(vals, bins=32)
        p = hist[hist > 0].astype(np.float64)
        p /= p.sum()
        ent = -np.sum(p * np.log2(p))
        entropy_ratio = float(np.clip(ent / np.log2(32.0), 0.0, 1.0))

    # Skewness / kurtosis (log-transformed)
    if std < EPS:
        skewness_t = 0.0
        kurtosis_excess_t = 0.0
    else:
        z = (vals - mean) / (std + EPS)
        skewness_t = _signed_log1p(float(np.mean(z**3)))
        kurtosis_excess_t = _signed_log1p(float(np.mean(z**4) - 3.0))

    # Longest run ratio
    longest_run_ratio = float(np.clip(longest_run / n, 0.0, 1.0))

    # IQR norm
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr_norm = float(np.clip((q75 - q25) / (value_range + EPS), 0.0, 1.0))

    # MAD norm
    mad_norm = float(np.clip(np.median(np.abs(vals - med)) / (value_range + EPS), 0.0, 1.0))

    # Top-k frequency ratios
    sorted_counts = np.sort(uniq_counts)[::-1]
    top1_freq_ratio = float(np.clip(sorted_counts[0] / n, 0.0, 1.0))
    top5_freq_ratio = float(np.clip(sorted_counts[:5].sum() / n, 0.0, 1.0))

    # Outlier ratio
    if std < EPS:
        outlier_ratio = 0.0
    else:
        z_abs = np.abs((vals - mean) / (std + EPS))
        outlier_ratio = float(np.clip(np.mean(z_abs > 3.0), 0.0, 1.0))

    # Mean abs diff norm
    if n > 1:
        mean_abs_diff_norm = float(np.clip(
            np.mean(np.abs(diffs)) / (value_range + EPS), 0.0, 1.0
        ))
    else:
        mean_abs_diff_norm = 0.0

    return {
        "length_norm": length_norm,
        "adj_sorted_ratio": adj_sorted_ratio,
        "duplicate_ratio": duplicate_ratio,
        "dispersion_ratio": dispersion_ratio,
        "runs_ratio": runs_ratio,
        "inversion_ratio": inversion_ratio,
        "entropy_ratio": entropy_ratio,
        "skewness_t": skewness_t,
        "kurtosis_excess_t": kurtosis_excess_t,
        "longest_run_ratio": longest_run_ratio,
        "iqr_norm": iqr_norm,
        "mad_norm": mad_norm,
        "top1_freq_ratio": top1_freq_ratio,
        "top5_freq_ratio": top5_freq_ratio,
        "outlier_ratio": outlier_ratio,
        "mean_abs_diff_norm": mean_abs_diff_norm,
    }


_FEATURE_NAMES = [
    "length_norm", "adj_sorted_ratio", "duplicate_ratio", "dispersion_ratio",
    "runs_ratio", "inversion_ratio", "entropy_ratio", "skewness_t",
    "kurtosis_excess_t", "longest_run_ratio", "iqr_norm", "mad_norm",
    "top1_freq_ratio", "top5_freq_ratio", "outlier_ratio", "mean_abs_diff_norm",
]


# ─── Sorting Implementations (ALL C-level) ───────────────────────────────

def sort_introsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="quicksort")

def sort_heapsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="heapsort")

def sort_timsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="stable")

def sort_counting(arr: np.ndarray) -> np.ndarray:
    if arr.size <= 1:
        return arr.copy()
    min_val = arr.min()
    max_val = arr.max()
    value_range = int(max_val - min_val) + 1
    if value_range > 100_000_000:
        # Fallback for huge ranges — counting sort not viable
        return np.sort(arr, kind="quicksort")
    shifted = (arr - min_val).astype(np.int64)
    counts = np.bincount(shifted, minlength=value_range)
    sorted_shifted = np.repeat(np.arange(value_range, dtype=np.int64), counts)
    return (sorted_shifted + min_val).astype(arr.dtype)


ALGORITHMS = {
    "introsort": sort_introsort,
    "heapsort":  sort_heapsort,
    "timsort":   sort_timsort,
    "counting_sort": sort_counting,
}


# ─── Timing ──────────────────────────────────────────────────────────────

def time_algorithm(func, arr: np.ndarray, repeats: int) -> float:
    """Time one sorting algorithm. Returns median of `repeats` runs (seconds)."""
    # Warmup (also verifies correctness on first call)
    result = func(arr.copy())

    times = []
    for _ in range(repeats):
        gc.disable()
        copy = arr.copy()
        start = time.perf_counter()
        _ = func(copy)
        elapsed = time.perf_counter() - start
        gc.enable()
        times.append(elapsed)

    return float(np.median(times))


def time_all_algorithms(arr: np.ndarray, repeats: int) -> dict[str, float]:
    """Time all 4 algorithms on the same array. Returns {name: median_seconds}."""
    timings = {}
    for name, func in ALGORITHMS.items():
        try:
            t = time_algorithm(func, arr, repeats)
            timings[name] = t
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}", file=sys.stderr)
            timings[name] = float("nan")
    return timings


# ─── Correctness Check (run once at startup) ─────────────────────────────

def verify_algorithms():
    """Verify all sort implementations produce correct results."""
    rng = np.random.default_rng(999)
    for n in [100, 10_000]:
        arr = rng.integers(0, 1_000_000, size=n, dtype=np.int32)
        expected = np.sort(arr)
        for name, func in ALGORITHMS.items():
            result = func(arr.copy())
            if not np.array_equal(result, expected):
                raise ValueError(f"CORRECTNESS FAILURE: {name} on n={n}")
    print("✓ All 4 algorithms verified correct.")


# ─── Split Logic ─────────────────────────────────────────────────────────

def assign_splits(
    records: list[dict],
) -> list[dict]:
    """
    Assign each record to a split:
      - uniform/normal → stratified into train/val/test_A (60/20/20)
      - lognormal/exponential → all go to test_B
    """
    rng = np.random.default_rng(SEED + 7)

    # Group in-distribution records by (size, distribution, structure)
    in_dist_groups: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        dist = rec["distribution"]
        if dist in TRAIN_DISTS:
            key = (rec["n"], dist, rec["structure"])
            in_dist_groups.setdefault(key, []).append(i)
        else:
            rec["split"] = "test_B"

    # Stratified split within each group
    for key, indices in sorted(in_dist_groups.items()):
        idx = np.array(indices)
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))
        # Remaining goes to test_A
        for j, i in enumerate(idx):
            if j < n_train:
                records[i]["split"] = "train"
            elif j < n_train + n_val:
                records[i]["split"] = "val"
            else:
                records[i]["split"] = "test_A"

    return records


# ─── Formatting ──────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ─── Main Pipeline ───────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(SIZES) * len(DISTRIBUTIONS) * len(STRUCTURES) * REPEATS
    print(f"Step 2: Full Timing Pipeline")
    print(f"============================")
    print(f"Sizes: {', '.join(_fmt(s) for s in SIZES)}")
    print(f"Distributions: {', '.join(DISTRIBUTIONS)}")
    print(f"Structures: {', '.join(STRUCTURES)}")
    print(f"Repeats: {REPEATS}")
    print(f"Total samples: {total}")
    print(f"Algorithms: {', '.join(ALGORITHMS.keys())}")
    print()

    # Verify correctness first
    verify_algorithms()

    # Determine n_max for length_norm normalization (max size in train distributions)
    n_max = float(max(SIZES))

    master_rng = np.random.default_rng(SEED)
    records: list[dict] = []

    t_pipeline_start = time.time()
    done = 0

    for size_idx, n in enumerate(SIZES):
        repeats = TIMING_REPEATS_SMALL if n <= 500_000 else TIMING_REPEATS_LARGE
        t_size_start = time.time()
        size_count = len(DISTRIBUTIONS) * len(STRUCTURES) * REPEATS

        print(f"\n── n={_fmt(n)} ({size_count} samples, {repeats} timing repeats) ──")

        for dist in DISTRIBUTIONS:
            for struct in STRUCTURES:
                for rep in range(REPEATS):
                    sample_id = f"s{done:06d}"
                    sample_seed = int(master_rng.integers(0, 2**31 - 1))
                    rng = np.random.default_rng(sample_seed)

                    # 1. Generate array
                    base = generate_base_array(rng, n, dist)
                    arr = apply_structure(rng, base, struct)

                    # 2. Extract features
                    feats = extract_features(arr, n_max=n_max, sample_id=sample_id)

                    # 3. Time all algorithms
                    timings = time_all_algorithms(arr, repeats=repeats)

                    # 4. Record
                    winner = min(timings, key=timings.get)
                    rec = {
                        "sample_id": sample_id,
                        "n": n,
                        "distribution": dist,
                        "structure": struct,
                        "repeat": rep,
                        "seed": sample_seed,
                        **feats,
                        "time_introsort": timings["introsort"],
                        "time_heapsort": timings["heapsort"],
                        "time_timsort": timings["timsort"],
                        "time_counting_sort": timings["counting_sort"],
                        "best_algorithm": winner,
                    }
                    records.append(rec)
                    done += 1

                    # Free memory explicitly for large arrays
                    del arr, base
                    gc.collect()

                # Progress for this (dist, struct) group
                tag = f"{dist}/{struct}"
                last_5 = records[-REPEATS:]
                winners = [r["best_algorithm"] for r in last_5]
                winner_str = max(set(winners), key=winners.count)
                print(f"  {tag:>25}: winner={winner_str:>14}  "
                      f"t_intro={last_5[-1]['time_introsort']*1000:7.2f}ms  "
                      f"t_heap={last_5[-1]['time_heapsort']*1000:7.2f}ms  "
                      f"t_tim={last_5[-1]['time_timsort']*1000:7.2f}ms  "
                      f"t_count={last_5[-1]['time_counting_sort']*1000:7.2f}ms")

        elapsed_size = time.time() - t_size_start
        elapsed_total = time.time() - t_pipeline_start
        remaining_sizes = len(SIZES) - size_idx - 1
        if size_idx > 0:
            avg_per_size = elapsed_total / (size_idx + 1)
            eta = avg_per_size * remaining_sizes
            print(f"  Time for n={_fmt(n)}: {elapsed_size:.1f}s  |  "
                  f"Total: {elapsed_total:.1f}s  |  ETA: {eta:.0f}s")

    # ── Assign splits ──
    records = assign_splits(records)

    # ── Build DataFrames ──
    df = pd.DataFrame(records)

    # Validate
    assert df["sample_id"].is_unique, "Duplicate sample IDs!"
    feat_cols = _FEATURE_NAMES
    feat_data = df[feat_cols].to_numpy()
    assert np.all(np.isfinite(feat_data)), "NaN/Inf in features!"
    time_cols = ["time_introsort", "time_heapsort", "time_timsort", "time_counting_sort"]
    time_data = df[time_cols].to_numpy()
    assert np.all(np.isfinite(time_data)), "NaN/Inf in timings!"
    assert np.all(time_data > 0), "Non-positive timings!"

    # ── Save per-split parquets ──
    split_counts = {}
    for split_name in ["train", "val", "test_A", "test_B"]:
        split_df = df[df["split"] == split_name].copy()
        split_df.to_parquet(out_dir / f"{split_name}.parquet", index=False)
        split_counts[split_name] = len(split_df)

    # Also save the complete dataset
    df.to_parquet(out_dir / "all_samples.parquet", index=False)

    # ── Save config ──
    config = {
        "seed": SEED,
        "sizes": SIZES,
        "distributions": DISTRIBUTIONS,
        "structures": STRUCTURES,
        "repeats": REPEATS,
        "timing_repeats_small": TIMING_REPEATS_SMALL,
        "timing_repeats_large": TIMING_REPEATS_LARGE,
        "algorithms": list(ALGORITHMS.keys()),
        "feature_set": "v2",
        "feature_names": _FEATURE_NAMES,
        "n_max": n_max,
        "total_samples": len(df),
        "split_counts": split_counts,
        "train_distributions": sorted(TRAIN_DISTS),
        "shift_distributions": sorted(SHIFT_DISTS),
    }
    (out_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2))

    # ── Summary statistics ──
    elapsed_total = time.time() - t_pipeline_start

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total samples: {len(df)}")
    print(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"\n  Splits:")
    for split_name, count in split_counts.items():
        print(f"    {split_name:>8}: {count:>4} samples")

    print(f"\n  Win counts (overall):")
    for algo in ALGORITHMS:
        cnt = int((df["best_algorithm"] == algo).sum())
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {algo:>14}: {cnt:>3} ({pct:4.1f}%)  {bar}")

    print(f"\n  Win counts by split:")
    for split_name in ["train", "val", "test_A", "test_B"]:
        split_df = df[df["split"] == split_name]
        if len(split_df) == 0:
            continue
        print(f"    {split_name}:")
        for algo in ALGORITHMS:
            cnt = int((split_df["best_algorithm"] == algo).sum())
            pct = cnt / len(split_df) * 100
            print(f"      {algo:>14}: {cnt:>3} ({pct:4.1f}%)")

    # VBS vs SBS sanity check
    vbs_total = df[time_cols].min(axis=1).sum()
    sbs_total = df["time_heapsort"].sum()
    gap = (sbs_total - vbs_total) / sbs_total * 100
    print(f"\n  VBS-SBS gap: {gap:.1f}% (heapsort wastes {gap:.1f}% of total time)")

    print(f"\n  Output: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    {f.name:>30}  {size_mb:.2f} MB")

    print(f"\n✓ Ready for Step 3 (XGBoost training)")


if __name__ == "__main__":
    main()
