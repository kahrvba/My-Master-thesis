#!/usr/bin/env python3
"""
VBS vs SBS Gap Analysis
========================
This is THE fundamental metric in algorithm selection (Rice 1976, AutoFolio, SATzilla).

- SBS (Single Best Solver) = "always pick heapsort" = the naive baseline
- VBS (Virtual Best Solver) = always pick the true fastest per instance = the ceiling
- Oracle selector = what a perfect model would achieve

If VBS is only 2% faster than SBS → thesis is pointless, just use heapsort.
If VBS is 20%+ faster than SBS → real value in selection, thesis is justified.

We measure:
  gap = (SBS_total_time - VBS_total_time) / SBS_total_time × 100%
  
Also show: per-instance penalty of "always heapsort" — the worst-case slowdown.
"""

from __future__ import annotations

import gc
import time

import numpy as np


# ── Same sorting implementations as pilot v2 ──────────────────────────────

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
        return np.sort(arr, kind="quicksort")
    shifted = (arr - min_val).astype(np.int64)
    counts = np.bincount(shifted, minlength=value_range)
    sorted_shifted = np.repeat(np.arange(value_range, dtype=np.int64), counts)
    return (sorted_shifted + min_val).astype(arr.dtype)


ALGORITHMS = {
    "introsort":     sort_introsort,
    "heapsort":      sort_heapsort,
    "timsort":       sort_timsort,
    "counting_sort": sort_counting,
}


# ── Timing ────────────────────────────────────────────────────────────────

def time_algorithm(func, arr: np.ndarray, repeats: int = 5) -> float:
    _ = func(arr.copy())  # warmup
    times = []
    for r in range(repeats):
        gc.disable()
        copy = arr.copy()
        start = time.perf_counter()
        _ = func(copy)
        elapsed = time.perf_counter() - start
        gc.enable()
        times.append(elapsed)
    return float(np.median(times))


# ── Generate test arrays ─────────────────────────────────────────────────

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


def generate_test_arrays() -> list[dict]:
    rng = np.random.default_rng(42)
    arrays = []

    # Focus on the "big data" regime the user cares about
    sizes = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    structures = {
        "random":          lambda arr, rng: rng.permutation(arr),
        "nearly_sorted":   lambda arr, rng: _nearly_sort(arr, rng),
        "reverse_sorted":  lambda arr, rng: arr[::-1].copy(),
        "few_unique":      lambda arr, rng: rng.choice(
            rng.integers(0, 16, size=16), size=arr.size
        ).astype(np.int32),
        "sorted_runs":     lambda arr, rng: _make_runs(arr, rng),
        "sorted":          lambda arr, rng: arr.copy(),
    }

    for n in sizes:
        base = rng.integers(0, 30_000_000, size=n, dtype=np.int32)
        base_sorted = np.sort(base)
        for struct_name, struct_fn in structures.items():
            if struct_name in ("few_unique",):
                arr = struct_fn(base, rng)
            elif struct_name in ("reverse_sorted", "nearly_sorted", "sorted"):
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


# ── Main Analysis ─────────────────────────────────────────────────────────

def main() -> None:
    test_cases = generate_test_arrays()
    algo_names = list(ALGORITHMS.keys())
    
    print(f"VBS vs SBS Gap Analysis")
    print(f"=======================")
    print(f"{len(test_cases)} test cases × {len(algo_names)} algorithms")
    print(f"Sizes: 100K to 10M  |  Structures: 6 patterns\n")

    # Collect all timings
    all_times = []  # list of dicts: {algo: time}

    header = f"{'test case':>30} | " + " | ".join(f"{name:>14}" for name in algo_names) + " | {'WINNER':>14} | {'heap penalty':>12}"
    print(header)
    print("-" * len(header))

    for tc in test_cases:
        arr = tc["arr"]
        n = tc["n"]
        repeats = 5 if n <= 1_000_000 else 3

        times = {}
        for name, func in ALGORITHMS.items():
            try:
                t = time_algorithm(func, arr, repeats=repeats)
                times[name] = t
            except Exception as e:
                times[name] = float("inf")

        best_algo = min(times, key=times.get)
        best_time = times[best_algo]
        heap_time = times["heapsort"]
        heap_penalty = (heap_time - best_time) / best_time * 100

        all_times.append({
            "name": tc["name"],
            "n": n,
            "structure": tc["structure"],
            "times": times,
            "winner": best_algo,
            "vbs_time": best_time,
            "sbs_time": heap_time,
            "heap_penalty_pct": heap_penalty,
        })

        # Format
        time_strs = []
        for name in algo_names:
            t = times[name]
            if name == best_algo:
                time_strs.append(f"{'*'+f'{t*1000:.1f}ms':>13}")
            else:
                time_strs.append(f"{f'{t*1000:.1f}ms':>14}")
        
        penalty_str = f"{heap_penalty:+.1f}%" if heap_penalty > 0.5 else "optimal"
        print(f"{tc['name']:>30} | " + " | ".join(time_strs) + f" | {best_algo:>14} | {penalty_str:>12}")

    # ── Aggregate Analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    total_vbs = sum(r["vbs_time"] for r in all_times)
    total_sbs = sum(r["sbs_time"] for r in all_times)
    gap = (total_sbs - total_vbs) / total_sbs * 100

    print(f"\n  Total time (VBS = perfect selector):  {total_vbs:.4f}s")
    print(f"  Total time (SBS = always heapsort):   {total_sbs:.4f}s")
    print(f"  Time wasted by 'always heapsort':     {total_sbs - total_vbs:.4f}s")
    print(f"")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  VBS-SBS GAP = {gap:.1f}%                        │")
    if gap < 5:
        print(f"  │  ⚠ Small gap — thesis may struggle          │")
    elif gap < 15:
        print(f"  │  ⚡ Moderate gap — thesis viable             │")
    else:
        print(f"  │  ✓ Large gap — strong thesis motivation      │")
    print(f"  └─────────────────────────────────────────────┘")

    # Win counts
    print(f"\n  WIN COUNTS:")
    win_counts = {}
    for r in all_times:
        w = r["winner"]
        win_counts[w] = win_counts.get(w, 0) + 1
    for name in algo_names:
        cnt = win_counts.get(name, 0)
        pct = cnt / len(all_times) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:>14}: {cnt:>2} wins ({pct:4.0f}%)  {bar}")

    # Per-instance penalty of always heapsort
    print(f"\n  WORST CASES FOR 'ALWAYS HEAPSORT':")
    print(f"  (instances where heapsort is most wasteful)\n")
    
    worst = sorted(all_times, key=lambda r: -r["heap_penalty_pct"])
    for r in worst[:10]:
        p = r["heap_penalty_pct"]
        if p > 0.5:
            print(f"    {r['name']:>30}:  heapsort is {p:+.1f}% slower than {r['winner']}")

    # Cases where heapsort is fastest
    print(f"\n  BEST CASES FOR 'ALWAYS HEAPSORT':")
    print(f"  (instances where heapsort is optimal)\n")
    
    best_for_heap = [r for r in all_times if r["winner"] == "heapsort"]
    for r in best_for_heap[:10]:
        # How much slower is the runner-up?
        times = r["times"]
        heap_t = times["heapsort"]
        others = {k: v for k, v in times.items() if k != "heapsort"}
        runner_up = min(others, key=others.get)
        runner_t = others[runner_up]
        advantage = (runner_t - heap_t) / heap_t * 100
        print(f"    {r['name']:>30}:  heapsort beats {runner_up} by {advantage:.1f}%")

    # Per-size breakdown
    print(f"\n  VBS-SBS GAP BY SIZE:")
    sizes = sorted(set(r["n"] for r in all_times))
    for sz in sizes:
        subset = [r for r in all_times if r["n"] == sz]
        vbs = sum(r["vbs_time"] for r in subset)
        sbs = sum(r["sbs_time"] for r in subset)
        g = (sbs - vbs) / sbs * 100
        print(f"    n={_fmt(sz):>4}:  VBS={vbs*1000:.1f}ms  SBS={sbs*1000:.1f}ms  gap={g:.1f}%")

    # Per-structure breakdown
    print(f"\n  VBS-SBS GAP BY DATA PATTERN:")
    structures = sorted(set(r["structure"] for r in all_times))
    for struct in structures:
        subset = [r for r in all_times if r["structure"] == struct]
        vbs = sum(r["vbs_time"] for r in subset)
        sbs = sum(r["sbs_time"] for r in subset)
        g = (sbs - vbs) / sbs * 100
        winner_freq = {}
        for r in subset:
            w = r["winner"]
            winner_freq[w] = winner_freq.get(w, 0) + 1
        top = max(winner_freq, key=winner_freq.get)
        print(f"    {struct:>15}:  gap={g:5.1f}%  dominant winner={top}")

    print(f"\n{'='*80}")
    print("CONCLUSION:")
    if gap < 5:
        print("  The gap is too small. 'Always heapsort' is nearly optimal.")
        print("  Consider: different algorithms, different data regimes, or reframe the thesis.")
    elif gap < 15:
        print("  Moderate gap. A selector adds real value in specific regimes.")
        print("  Thesis is viable but needs to emphasize WHERE selection helps most.")
    else:
        print(f"  Strong gap of {gap:.1f}%. 'Always heapsort' wastes significant time.")
        print("  A selector has clear engineering and academic value.")
        print("  This is publishable algorithm selection territory.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
