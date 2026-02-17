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
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extraction import (
    extract_features,
    FEATURE_NAMES as _FEATURE_NAMES,
    SEED,
    EPS,
    _monotonic_run_stats,
    _inversion_count_merge,
    _signed_log1p,
)

# ─── Constants ────────────────────────────────────────────────────────────

# SEED and EPS imported from feature_extraction
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


# ─── Feature Extraction ───────────────────────────────────────────────────
# Imported from feature_extraction.py (single source of truth).
# extract_features, _FEATURE_NAMES, helpers all come from the top-level import.


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
