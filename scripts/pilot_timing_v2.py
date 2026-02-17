#!/usr/bin/env python3
"""
Pilot timing v2: fix the problems found in v1.

Problems fixed:
1. Array sizes scaled up: 10K, 100K, 500K, 1M, 5M, 10M
2. Radix sort rewritten: fully C-level via np.argsort on digit keys
3. Counting sort: already C-level but now offset-aware for memory
4. Measure feature extraction cost at larger scales

Goal: verify different algorithms win in different regions.
If one still dominates >80%, we need to rethink further.
"""

from __future__ import annotations

import gc
import time

import numpy as np


# ── Sorting implementations (ALL C-level) ─────────────────────────────────

def sort_introsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="quicksort")

def sort_heapsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="heapsort")

def sort_timsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="stable")

def sort_python_timsort(arr: np.ndarray) -> np.ndarray:
    return np.array(sorted(arr.tolist()), dtype=arr.dtype)

def sort_radix_lsd(arr: np.ndarray) -> np.ndarray:
    """
    LSD Radix sort — FULLY C-LEVEL.
    Uses np.argsort(digits, kind='stable') for each digit pass.
    No Python loops over elements.
    """
    if arr.size <= 1:
        return arr.copy()

    min_val = arr.min()
    shifted = (arr - min_val).astype(np.uint64)
    max_val = int(shifted.max())

    if max_val == 0:
        return arr.copy()

    radix_bits = 8
    mask = (1 << radix_bits) - 1

    # Count number of passes needed
    num_passes = 0
    test = max_val
    while test > 0:
        num_passes += 1
        test >>= radix_bits

    output = shifted.copy()
    for pass_num in range(num_passes):
        shift = pass_num * radix_bits
        digits = ((output >> shift) & mask).astype(np.int64)
        # Stable argsort on digits — C-level, O(n) for small key space
        order = np.argsort(digits, kind="stable")
        output = output[order]

    return (output.astype(arr.dtype) + min_val)


def sort_counting(arr: np.ndarray) -> np.ndarray:
    """
    Counting sort — fully C-level via np.bincount + np.repeat.
    Offsets values to minimize memory.
    """
    if arr.size <= 1:
        return arr.copy()

    min_val = arr.min()
    max_val = arr.max()
    value_range = int(max_val - min_val) + 1

    # Skip if value range is absurdly large (> 100M) — memory unsafe
    if value_range > 100_000_000:
        # Fallback: just return introsort (this algorithm is not viable here)
        return np.sort(arr, kind="quicksort")

    shifted = (arr - min_val).astype(np.int64)
    counts = np.bincount(shifted, minlength=value_range)
    sorted_shifted = np.repeat(np.arange(value_range, dtype=np.int64), counts)
    return (sorted_shifted + min_val).astype(arr.dtype)


ALGORITHMS = {
    "introsort":       sort_introsort,
    "heapsort":        sort_heapsort,
    "timsort":         sort_timsort,
    "python_timsort":  sort_python_timsort,
    "radix_lsd":       sort_radix_lsd,
    "counting_sort":   sort_counting,
}


# ── Timing ────────────────────────────────────────────────────────────────

def time_algorithm(func, arr: np.ndarray, repeats: int = 5) -> float:
    _ = func(arr.copy())  # warmup

    times = []
    for r in range(repeats):
        gc.disable()
        copy = arr.copy()
        start = time.perf_counter()
        result = func(copy)
        elapsed = time.perf_counter() - start
        gc.enable()
        times.append(elapsed)

        if r == 0:
            expected = np.sort(arr)
            if not np.array_equal(result, expected):
                raise ValueError(f"{func.__name__} incorrect on size {arr.size}!")

    return float(np.median(times))


# ── Generate test arrays directly (no file dependency) ────────────────────

def generate_test_arrays() -> list[dict]:
    """Generate arrays covering the full design space at scale."""
    rng = np.random.default_rng(42)
    arrays = []

    sizes = [10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    structures = {
        "random":        lambda arr, rng: rng.permutation(arr),
        "nearly_sorted": lambda arr, rng: _nearly_sort(arr, rng),
        "reverse_sorted": lambda arr, rng: arr[::-1].copy(),
        "few_unique":    lambda arr, rng: rng.choice(rng.integers(0, 16, size=16), size=arr.size).astype(np.int32),
        "runs":          lambda arr, rng: _make_runs(arr, rng),
    }

    for n in sizes:
        base = rng.integers(0, 30_000_000, size=n, dtype=np.int32)
        base_sorted = np.sort(base)
        for struct_name, struct_fn in structures.items():
            if struct_name in ("few_unique",):
                arr = struct_fn(base, rng)
            elif struct_name in ("reverse_sorted", "nearly_sorted"):
                arr = struct_fn(base_sorted, rng)
            else:
                arr = struct_fn(base, rng)
            arrays.append({
                "name": f"n={_fmt(n)}_{struct_name}",
                "n": n,
                "structure": struct_name,
                "arr": arr,
            })

    return arrays


def _nearly_sort(arr, rng):
    out = arr.copy()
    n = out.size
    swaps = max(1, int(0.02 * n))
    i_idx = rng.integers(0, n, size=swaps)
    j_idx = rng.integers(0, n, size=swaps)
    out[i_idx], out[j_idx] = out[j_idx].copy(), out[i_idx].copy()
    return out


def _make_runs(arr, rng):
    out = rng.permutation(arr)
    n = out.size
    chunks = []
    i = 0
    while i < n:
        run_len = min(int(rng.integers(64, 512)), n - i)
        chunk = np.sort(out[i:i + run_len])
        if rng.random() < 0.5:
            chunk = chunk[::-1]
        chunks.append(chunk)
        i += run_len
    return np.concatenate(chunks)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    test_cases = generate_test_arrays()
    algo_names = list(ALGORITHMS.keys())

    print(f"Pilot v2: {len(test_cases)} test cases × {len(algo_names)} algorithms")
    print(f"Array sizes: 10K to 10M\n")

    header = f"{'test case':>30} | " + " | ".join(f"{name:>15}" for name in algo_names) + " | WINNER"
    print(header)
    print("-" * len(header))

    win_counts: dict[str, int] = {name: 0 for name in algo_names}
    results = []

    for tc in test_cases:
        arr = tc["arr"]
        n = tc["n"]

        # Reduce repeats for very large arrays to keep runtime reasonable
        repeats = 5 if n <= 1_000_000 else 3

        times = {}
        for name, func in ALGORITHMS.items():
            # Skip python_timsort for very large arrays (too slow)
            if name == "python_timsort" and n > 1_000_000:
                times[name] = float("inf")
                continue
            # Skip counting_sort if value range would be huge
            if name == "counting_sort" and tc["structure"] != "few_unique":
                # For random data with range 0-30M, counting sort allocates 30M array
                # Let it try — it will be slow but won't crash
                pass

            try:
                t = time_algorithm(func, arr, repeats=repeats)
                times[name] = t
            except Exception as e:
                times[name] = float("inf")
                print(f"  WARN: {name} failed on {tc['name']}: {e}")

        winner = min(times, key=times.get)
        win_counts[winner] += 1

        # Format output
        time_strs = []
        for name in algo_names:
            t = times[name]
            if t == float("inf"):
                time_strs.append(f"{'skip':>15}")
            elif name == winner:
                time_strs.append(f"{'*' + f'{t*1000:.2f}ms':>14}")
            else:
                ratio = t / times[winner]
                time_strs.append(f"{f'{t*1000:.2f}ms':>10}{f'({ratio:.1f}x)':>5}")

        print(f"{tc['name']:>30} | " + " | ".join(time_strs) + f" | {winner}")
        results.append({"name": tc["name"], "n": n, "structure": tc["structure"],
                         "winner": winner, **times})

    # Summary
    print("\n" + "=" * 70)
    print("WIN COUNTS:")
    total = len(test_cases)
    for name in algo_names:
        bar = "█" * (win_counts[name] * 2)
        pct = win_counts[name] / total * 100
        print(f"  {name:>15}: {win_counts[name]:>3} ({pct:4.0f}%)  {bar}")

    dominant = max(win_counts, key=win_counts.get)
    dominant_pct = win_counts[dominant] / total * 100

    if dominant_pct > 80:
        print(f"\n⚠  {dominant} dominates {dominant_pct:.0f}% — selection problem still trivial.")
    elif dominant_pct > 60:
        print(f"\n⚡ {dominant} leads at {dominant_pct:.0f}% but others have regions.")
    else:
        print(f"\n✓  Good diversity — no algorithm dominates.")

    # Where does each algorithm win?
    print("\n" + "=" * 70)
    print("WHERE EACH ALGORITHM WINS:")
    for name in algo_names:
        wins = [r for r in results if r["winner"] == name]
        if wins:
            regions = [f"{w['name']}" for w in wins]
            print(f"  {name}: {', '.join(regions)}")

    # Feature extraction cost at scale
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COST vs SORTING:")
    for n_test in [10_000, 100_000, 1_000_000, 10_000_000]:
        arr = np.random.default_rng(99).integers(0, 30_000_000, size=n_test, dtype=np.int32).astype(np.float64)

        # Simulate v2 feature extraction cost (main expensive ops)
        gc.disable()
        start = time.perf_counter()
        _ = float(arr.size)
        _ = np.diff(arr)
        _ = np.unique(arr)
        _ = float(arr.std())
        _ = float(arr.min())
        _ = float(arr.max())
        _ = np.percentile(arr, [25, 75])
        _ = np.median(arr)
        _ = np.median(np.abs(arr - np.median(arr)))
        _ = np.histogram(arr, bins=32)
        _ = np.mean(np.abs(np.diff(arr)))
        feat_time = time.perf_counter() - start
        gc.enable()

        # Fastest sort at this size
        gc.disable()
        copy = arr.astype(np.int32)
        start = time.perf_counter()
        _ = np.sort(copy, kind="quicksort")
        sort_time = time.perf_counter() - start
        gc.enable()

        ratio = feat_time / sort_time
        print(f"  n={_fmt(n_test):>4}: features={feat_time*1000:.2f}ms  sort={sort_time*1000:.2f}ms  ratio={ratio:.2f}x  {'✓ features cheaper' if ratio < 1 else '✗ features costlier'}")


if __name__ == "__main__":
    main()
