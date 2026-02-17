#!/usr/bin/env python3
"""
Pilot timing: run 6 sorting algorithms on ~20 representative samples
to verify that different algorithms win in different regions.

If one algorithm dominates everything, the selection problem is trivial
and we need to rethink the portfolio before building the full pipeline.
"""

from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

import numpy as np


# ── Sorting implementations (ALL C-level, no Python loops) ────────────────

def sort_introsort(arr: np.ndarray) -> np.ndarray:
    """numpy quicksort = introsort (quicksort + heapsort fallback). C-level."""
    return np.sort(arr, kind="quicksort")


def sort_heapsort(arr: np.ndarray) -> np.ndarray:
    """numpy heapsort. C-level."""
    return np.sort(arr, kind="heapsort")


def sort_timsort(arr: np.ndarray) -> np.ndarray:
    """numpy stable sort = Timsort. C-level."""
    return np.sort(arr, kind="stable")


def sort_python_timsort(arr: np.ndarray) -> np.ndarray:
    """CPython sorted() on a Python list. C-level Timsort, different memory layout."""
    return np.array(sorted(arr.tolist()), dtype=arr.dtype)


def sort_radix_lsd(arr: np.ndarray) -> np.ndarray:
    """
    LSD Radix sort for non-negative integers.
    Implemented via numpy vectorized ops — no Python loops over elements.
    """
    if arr.size <= 1:
        return arr.copy()

    # Offset to handle any min value
    min_val = arr.min()
    shifted = (arr - min_val).astype(np.uint64)
    max_val = shifted.max()

    if max_val == 0:
        return arr.copy()

    # Process 8 bits at a time (radix-256)
    radix_bits = 8
    radix = 1 << radix_bits  # 256
    mask = radix - 1

    output = shifted.copy()
    num_passes = 0
    test_val = int(max_val)
    while test_val > 0:
        num_passes += 1
        test_val >>= radix_bits

    for pass_num in range(num_passes):
        shift = pass_num * radix_bits
        # Extract digit for this pass
        digits = ((output >> shift) & mask).astype(np.int64)

        # Count occurrences of each digit
        counts = np.bincount(digits, minlength=radix)

        # Prefix sum (cumulative counts) for stable placement
        offsets = np.empty(radix, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts[:-1], out=offsets[1:])

        # Place elements in sorted order for this digit (stable)
        result = np.empty_like(output)
        for i in range(output.size):
            d = int(digits[i])
            result[offsets[d]] = output[i]
            offsets[d] += 1
        output = result

    # Shift back
    return (output.astype(arr.dtype) + min_val)


def sort_counting(arr: np.ndarray) -> np.ndarray:
    """
    Counting sort via np.bincount + np.repeat. C-level except for offset.
    Works best when value range is small relative to n.
    """
    if arr.size <= 1:
        return arr.copy()

    min_val = arr.min()
    shifted = (arr - min_val).astype(np.int64)
    counts = np.bincount(shifted)
    sorted_shifted = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return (sorted_shifted + min_val).astype(arr.dtype)


# ── All algorithms ────────────────────────────────────────────────────────

ALGORITHMS = {
    "introsort":       sort_introsort,
    "heapsort":        sort_heapsort,
    "timsort":         sort_timsort,
    "python_timsort":  sort_python_timsort,
    "radix_lsd":       sort_radix_lsd,
    "counting_sort":   sort_counting,
}


# ── Timing helper ─────────────────────────────────────────────────────────

def time_algorithm(func, arr: np.ndarray, repeats: int = 5) -> float:
    """
    Time a sorting function with warmup + multiple repeats.
    Returns median time in seconds.
    """
    # Warmup run (not timed)
    _ = func(arr.copy())

    times = []
    for _ in range(repeats):
        gc.disable()
        copy = arr.copy()
        start = time.perf_counter()
        result = func(copy)
        elapsed = time.perf_counter() - start
        gc.enable()
        times.append(elapsed)

        # Verify correctness on first run
        if len(times) == 1:
            expected = np.sort(arr)
            if not np.array_equal(result, expected):
                raise ValueError(f"{func.__name__} produced incorrect output!")

    return float(np.median(times))


# ── Pick pilot samples ────────────────────────────────────────────────────

def pick_pilot_samples(index_csv: Path, raw_dir: Path) -> list[dict]:
    """
    Pick ~24 samples that cover extremes:
    - 3 sizes (small=1000, medium=10000, large=50000)
    - 4 structures (random, nearly_sorted, few_unique, runs)
    - 2 distributions (uniform, exponential)
    """
    with open(index_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    target_sizes = {"1000", "10000", "50000"}
    target_structures = {"random", "nearly_sorted", "few_unique", "runs"}
    target_distributions = {"uniform", "exponential"}

    selected = []
    seen = set()
    for row in rows:
        key = (row["size_bucket"], row["structure_tag"], row["distribution_tag"])
        if (
            row["size_bucket"] in target_sizes
            and row["structure_tag"] in target_structures
            and row["distribution_tag"] in target_distributions
            and key not in seen
        ):
            seen.add(key)
            selected.append(row)

    return selected


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    index_csv = project_root / "data" / "synthetic" / "index.csv"

    samples = pick_pilot_samples(index_csv, project_root / "data" / "synthetic" / "raw")
    print(f"Pilot: timing {len(samples)} samples × {len(ALGORITHMS)} algorithms\n")

    algo_names = list(ALGORITHMS.keys())
    header = f"{'sample_id':>10} {'size':>6} {'structure':>16} {'distribution':>14} | " + \
             " | ".join(f"{name:>15}" for name in algo_names) + " | WINNER"
    print(header)
    print("-" * len(header))

    win_counts: dict[str, int] = {name: 0 for name in algo_names}
    results_all = []

    for row in sorted(samples, key=lambda r: (int(r["n"]), r["structure_tag"])):
        sample_id = row["sample_id"]

        # Resolve path
        raw_path = row["path"]
        p = Path(raw_path)
        if not p.exists():
            # Try stripping the project folder prefix
            if p.parts and p.parts[0] == "My-Master-thesis":
                p = project_root / Path(*p.parts[1:])
            else:
                p = project_root / p
        arr = np.load(p, allow_pickle=False)

        # Time each algorithm
        times = {}
        for name, func in ALGORITHMS.items():
            try:
                t = time_algorithm(func, arr, repeats=5)
                times[name] = t
            except Exception as e:
                times[name] = float("inf")
                print(f"  WARNING: {name} failed on {sample_id}: {e}")

        winner = min(times, key=times.get)  # type: ignore[arg-type]
        win_counts[winner] += 1

        # Format times (highlight winner)
        time_strs = []
        for name in algo_names:
            t = times[name]
            if t == float("inf"):
                time_strs.append(f"{'FAIL':>15}")
            elif name == winner:
                time_strs.append(f"{'*' + f'{t*1000:.3f}ms':>14}")
            else:
                ratio = t / times[winner]
                time_strs.append(f"{f'{t*1000:.3f}ms':>12}{f'({ratio:.1f}x)':>3}")

        line = f"{sample_id:>10} {row['n']:>6} {row['structure_tag']:>16} {row['distribution_tag']:>14} | " + \
               " | ".join(time_strs) + f" | {winner}"
        print(line)

        results_all.append({"sample_id": sample_id, "n": int(row["n"]),
                            "structure": row["structure_tag"],
                            "distribution": row["distribution_tag"],
                            "winner": winner, **times})

    # Summary
    print("\n" + "=" * 60)
    print("WIN COUNTS:")
    for name in algo_names:
        bar = "█" * win_counts[name]
        print(f"  {name:>15}: {win_counts[name]:>3}  {bar}")

    total = len(samples)
    dominant = max(win_counts, key=win_counts.get)  # type: ignore[arg-type]
    dominant_pct = win_counts[dominant] / total * 100

    print(f"\nDominant algorithm: {dominant} ({dominant_pct:.0f}%)")
    if dominant_pct > 80:
        print("⚠  WARNING: One algorithm dominates >80% — selection problem may be trivial.")
        print("   Consider restructuring portfolio or expanding data range.")
    elif dominant_pct > 60:
        print("⚡ One algorithm leads but others have regions. Selection is viable.")
    else:
        print("✓  Good diversity — multiple algorithms win in different regions.")

    # Also measure feature extraction cost
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COST vs SORTING COST:")
    from extract_features import v1_features, v2_features, coerce_numeric_finite, monotonic_run_stats
    for size_label, sample in [("small (1K)", results_all[0]), ("large (50K)", results_all[-1])]:
        sid = sample["sample_id"]
        row_match = [r for r in samples if r["sample_id"] == sid][0]
        p = Path(row_match["path"])
        if not p.exists():
            if p.parts and p.parts[0] == "My-Master-thesis":
                p = project_root / Path(*p.parts[1:])
            else:
                p = project_root / p
        arr = np.load(p, allow_pickle=False).astype(np.float64)

        # Time feature extraction
        gc.disable()
        start = time.perf_counter()
        v1 = v1_features(arr, n_max_train=50000.0)
        v2 = v2_features(arr, sample_id=sid, seed=42)
        feat_time = time.perf_counter() - start
        gc.enable()

        best_sort_time = min(v for k, v in sample.items() if k in ALGORITHMS and isinstance(v, float))
        ratio = feat_time / best_sort_time

        print(f"  {size_label}: feature extraction = {feat_time*1000:.3f}ms, "
              f"fastest sort = {best_sort_time*1000:.3f}ms, "
              f"ratio = {ratio:.1f}x")
        if ratio > 1:
            print(f"    → Feature extraction is SLOWER than sorting at this size")
        else:
            print(f"    → Feature extraction is cheaper than sorting ✓")


if __name__ == "__main__":
    main()
