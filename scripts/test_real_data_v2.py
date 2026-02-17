#!/usr/bin/env python3
"""
Real-World Feature Extraction Test v2 — Full-Race F1 Telemetry (Large Arrays)
===============================================================================
v1 used only the fastest lap per driver (~700 samples).
v2 concatenates ALL laps across the full race to produce realistic large-scale
arrays (10K–50K+ samples) — matching our benchmark's target range (10K–2M).

Also tests multiple GPs and sessions for diversity.

Purpose: Prove that feature extraction + algorithm selection is valid at scale
         on real-world data, not just toy-size single-lap telemetry.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

# ── Import our feature extraction ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_algorithms import extract_features, _FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "f1_cache"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "all_samples.parquet"
V1_PATH = PROJECT_ROOT / "data" / "real_world" / "f1_real_world_results.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_world_v2"
N_MAX = 2_000_000.0

# Sessions to test — diversity of tracks and conditions
SESSIONS = [
    (2024, "Bahrain", "R"),      # Desert, long straights
    (2024, "Monaco", "R"),       # Tight street circuit
    (2024, "Monza", "R"),        # High-speed temple
]

CHANNELS = ["Speed", "RPM", "Throttle", "Brake", "nGear", "DRS", "Distance"]

ALGORITHMS = {
    "introsort": lambda a: np.sort(a, kind="quicksort"),
    "heapsort":  lambda a: np.sort(a, kind="heapsort"),
    "timsort":   lambda a: np.sort(a, kind="stable"),
}


def fetch_session(year: int, gp: str, session_type: str):
    """Load an F1 session with telemetry."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def extract_large_arrays(session, gp_label: str) -> list[tuple[str, np.ndarray]]:
    """
    Extract LARGE arrays by concatenating telemetry across ALL laps.

    Strategies:
      1. Full-race concat per channel: all laps of one driver → one big array
      2. Multi-driver concat: all drivers' speed data → massive array
      3. Single large laps: drivers with the most telemetry points
    """
    arrays = []
    event_name = gp_label

    # ── Strategy 1: Full-race per driver (top 5 drivers) ─────────────────
    for drv in session.drivers[:5]:
        try:
            laps = session.laps.pick_drivers(drv)
            if len(laps) == 0:
                continue
            driver_name = laps.iloc[0]["Driver"]

            # Collect telemetry from ALL laps
            all_tel = []
            for _, lap in laps.iterrows():
                try:
                    tel = lap.get_telemetry()
                    if tel is not None and len(tel) > 0:
                        all_tel.append(tel)
                except Exception:
                    continue

            if not all_tel:
                continue

            combined = pd.concat(all_tel, ignore_index=True)
            n_laps = len(all_tel)

            for ch in CHANNELS:
                if ch in combined.columns:
                    vals = combined[ch].dropna().to_numpy(dtype=np.float64)
                    if len(vals) > 1000:  # Only keep arrays > 1K
                        label = f"{event_name}_{driver_name}_{ch.lower()}_full_race"
                        arrays.append((f"{label} (n={len(vals)}, {n_laps}laps)", vals))

        except Exception as e:
            print(f"  ⚠ Skipping driver {drv}: {e}")
            continue

    # ── Strategy 2: All-drivers concat for speed (massive array) ─────────
    try:
        all_speed = []
        for drv in session.drivers:
            try:
                laps = session.laps.pick_drivers(drv)
                for _, lap in laps.iterrows():
                    try:
                        tel = lap.get_telemetry()
                        if tel is not None and "Speed" in tel.columns:
                            all_speed.append(tel["Speed"].dropna().to_numpy(dtype=np.float64))
                    except Exception:
                        continue
            except Exception:
                continue
        if all_speed:
            mega_speed = np.concatenate(all_speed)
            if len(mega_speed) > 10000:
                arrays.append((f"{event_name}_ALL_drivers_speed (n={len(mega_speed)})", mega_speed))
    except Exception as e:
        print(f"  ⚠ Skipping all-drivers concat: {e}")

    return arrays


def time_algorithms(arr: np.ndarray, n_repeats: int = 3) -> dict[str, float]:
    """Time each algorithm, take median of n_repeats."""
    timings = {}
    for algo_name, algo_fn in ALGORITHMS.items():
        times_list = []
        for _ in range(n_repeats):
            copy = arr.copy()
            t0 = time.perf_counter()
            algo_fn(copy)
            t1 = time.perf_counter()
            times_list.append(t1 - t0)
        timings[f"time_{algo_name}"] = float(np.median(times_list))
    return timings


def main():
    print("=" * 110)
    print("REAL-WORLD FEATURE EXTRACTION TEST v2 — FULL-RACE F1 TELEMETRY (LARGE ARRAYS)")
    print("=" * 110)
    print(f"\nv1 tested ~700 samples per array (1 fastest lap)")
    print(f"v2 concatenates ALL laps → 10K–500K+ samples per array\n")

    all_arrays: list[tuple[str, np.ndarray]] = []

    # ── Fetch multiple sessions ───────────────────────────────────────────
    for year, gp, stype in SESSIONS:
        print(f"\n▸ Fetching {year} {gp} {stype}...")
        try:
            session = fetch_session(year, gp, stype)
            gp_label = f"{year}_{gp}"
            arrays = extract_large_arrays(session, gp_label)
            print(f"  ✓ Extracted {len(arrays)} arrays from {session.event['EventName']}")
            all_arrays.extend(arrays)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    if not all_arrays:
        print("  ✗ No arrays extracted — exiting")
        sys.exit(1)

    # Sort by size descending for display
    all_arrays.sort(key=lambda x: len(x[1]), reverse=True)

    # ── Size summary ──────────────────────────────────────────────────────
    sizes = [len(a) for _, a in all_arrays]
    print(f"\n{'=' * 110}")
    print(f"ARRAY SIZE SUMMARY: {len(all_arrays)} arrays")
    print(f"  Min: {min(sizes):,}  |  Median: {int(np.median(sizes)):,}  |  Max: {max(sizes):,}")
    print(f"  Arrays > 10K: {sum(1 for s in sizes if s > 10_000)}  |  > 50K: {sum(1 for s in sizes if s > 50_000)}  |  > 100K: {sum(1 for s in sizes if s > 100_000)}")
    print(f"{'=' * 110}")

    # ── Feature extraction + timing ───────────────────────────────────────
    print(f"\n▸ Running feature extraction + algorithm timing...")
    print(f"{'─' * 130}")
    print(f"{'Array':<65} {'n':>8} │ {'adj_sort':>8} {'dup_rat':>8} {'inv_rat':>8} {'entropy':>8} {'runs':>8} │ {'winner':>10} {'t_win':>10}")
    print(f"{'─' * 130}")

    rows = []
    for desc, arr in all_arrays:
        f = extract_features(arr, n_max=N_MAX, sample_id=desc)
        timings = time_algorithms(arr)

        winner = min(timings, key=timings.get).replace("time_", "")
        best_time = min(timings.values())
        row = {"array": desc, "n": len(arr), **f, **timings, "best_algorithm": winner}
        rows.append(row)

        print(f"{desc:<65} {len(arr):>8} │ "
              f"{f['adj_sorted_ratio']:>8.4f} "
              f"{f['duplicate_ratio']:>8.4f} "
              f"{f['inversion_ratio']:>8.4f} "
              f"{f['entropy_ratio']:>8.4f} "
              f"{f['runs_ratio']:>8.4f} │ "
              f"{winner:>10} {best_time:>10.6f}")

    real_df = pd.DataFrame(rows)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("SANITY CHECKS")
    print(f"{'=' * 110}")

    issues = 0

    feat_vals = real_df[_FEATURE_NAMES].to_numpy()
    nan_count = int(np.sum(np.isnan(feat_vals)))
    inf_count = int(np.sum(np.isinf(feat_vals)))
    if nan_count > 0 or inf_count > 0:
        print(f"  ✗ NaN: {nan_count}, Inf: {inf_count}")
        issues += 1
    else:
        print(f"  ✓ No NaN or Inf in any feature")

    bounded = [ft for ft in _FEATURE_NAMES if ft not in ("skewness_t", "kurtosis_excess_t")]
    bound_violations = []
    for feat in bounded:
        mn, mx = float(real_df[feat].min()), float(real_df[feat].max())
        if mn < -1e-9 or mx > 1.0 + 1e-9:
            bound_violations.append((feat, mn, mx))
            issues += 1
    if bound_violations:
        for feat, mn, mx in bound_violations:
            print(f"  ✗ {feat} out of bounds: [{mn:.6f}, {mx:.6f}]")
    else:
        print(f"  ✓ All bounded features in [0, 1]")

    # Distance arrays should be sorted
    for _, row in real_df.iterrows():
        if "distance" in row["array"]:
            asr = row["adj_sorted_ratio"]
            status = "✓" if asr > 0.95 else "✗"
            print(f"  {status} Distance: adj_sorted={asr:.4f}")
            if asr <= 0.95:
                issues += 1

    # Gear arrays should have high duplicate_ratio
    for _, row in real_df.iterrows():
        if "_gear_" in row["array"] or "gear" in row["array"]:
            dr = row["duplicate_ratio"]
            status = "✓" if dr > 0.95 else "✗"
            print(f"  {status} Gear: dup_ratio={dr:.4f}")
            if dr <= 0.95:
                issues += 1

    if issues == 0:
        print(f"\n  ✓ ALL SANITY CHECKS PASSED")
    else:
        print(f"\n  ⚠ {issues} issues found")

    # ── Feature comparison vs benchmark ───────────────────────────────────
    print(f"\n{'=' * 110}")
    print("FEATURE COMPARISON: v2 Real Data vs Synthetic Benchmark")
    print(f"{'=' * 110}")

    if BENCHMARK_PATH.exists():
        bench = pd.read_parquet(BENCHMARK_PATH)[_FEATURE_NAMES]
        print(f"\nBenchmark: {len(bench)} samples | Real v2: {len(real_df)} arrays\n")
        print(f"{'Feature':<24} {'Bench min':>10} {'Bench max':>10} │ {'Real min':>10} {'Real max':>10} │ {'Status':>10}")
        print("─" * 100)
        for feat in _FEATURE_NAMES:
            b_min, b_max = float(bench[feat].min()), float(bench[feat].max())
            r_min, r_max = float(real_df[feat].min()), float(real_df[feat].max())
            in_range = "✓" if r_min >= b_min - 0.1 and r_max <= b_max + 0.1 else "⚠ OUTSIDE"
            print(f"{feat:<24} {b_min:>10.4f} {b_max:>10.4f} │ {r_min:>10.4f} {r_max:>10.4f} │ {in_range:>10}")

    # ── v1 vs v2 comparison ───────────────────────────────────────────────
    if V1_PATH.exists():
        print(f"\n{'=' * 110}")
        print("v1 vs v2 COMPARISON")
        print(f"{'=' * 110}")
        v1 = pd.read_parquet(V1_PATH)
        print(f"\n  v1: {len(v1)} arrays, sizes {int(v1['n'].min())}–{int(v1['n'].max())}")
        print(f"  v2: {len(real_df)} arrays, sizes {int(real_df['n'].min()):,}–{int(real_df['n'].max()):,}")

        time_cols = [c for c in real_df.columns if c.startswith("time_")]

        # v1 winners
        v1_winners = v1["best_algorithm"].value_counts()
        print(f"\n  v1 winners: {v1_winners.to_dict()}")
        print(f"  v2 winners: {real_df['best_algorithm'].value_counts().to_dict()}")

        # v1 VBS-SBS
        if all(c in v1.columns for c in time_cols):
            v1_tm = v1[time_cols].to_numpy()
            v1_vbs = v1_tm.min(axis=1).sum()
            v1_sbs_idx = v1_tm.sum(axis=0).argmin()
            v1_sbs = v1_tm[:, v1_sbs_idx].sum()
            v1_gap = (v1_sbs - v1_vbs) / v1_vbs * 100
            print(f"\n  v1 VBS-SBS gap: {v1_gap:.1f}%  (n ≈ 700)")

    # ── Algorithm timing summary ──────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("ALGORITHM TIMING SUMMARY (v2 — Full Race)")
    print(f"{'=' * 110}")

    winner_counts = real_df["best_algorithm"].value_counts()
    total = len(real_df)
    print(f"\n  Win counts ({total} arrays):")
    for algo, count in winner_counts.items():
        print(f"    {algo:<12} {count:>3} wins  ({100*count/total:.1f}%)")

    time_cols = [c for c in real_df.columns if c.startswith("time_")]
    time_matrix = real_df[time_cols].to_numpy()
    vbs = time_matrix.min(axis=1).sum()
    sbs_algo = time_matrix.sum(axis=0).argmin()
    sbs = time_matrix[:, sbs_algo].sum()
    gap = (sbs - vbs) / vbs * 100
    sbs_name = time_cols[sbs_algo].replace("time_", "")
    print(f"\n  VBS total: {vbs:.6f}s")
    print(f"  SBS total: {sbs:.6f}s (always {sbs_name})")
    print(f"  VBS-SBS gap: {gap:.1f}%")

    # Per-array timing
    print(f"\n{'─' * 130}")
    print(f"{'Array':<65} {'introsort':>12} {'heapsort':>12} {'timsort':>12} │ {'winner':>10} {'margin':>8}")
    print(f"{'─' * 130}")
    for _, row in real_df.iterrows():
        ts = {a: row[f"time_{a}"] for a in ALGORITHMS}
        best_t = min(ts.values())
        second_t = sorted(ts.values())[1]
        margin = (second_t - best_t) / (best_t + 1e-15) * 100
        print(f"{row['array']:<65} "
              f"{ts['introsort']:>12.6f} {ts['heapsort']:>12.6f} {ts['timsort']:>12.6f} │ "
              f"{row['best_algorithm']:>10} {margin:>7.1f}%")

    # ── Breakdown by array size bucket ────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("WIN RATES BY ARRAY SIZE BUCKET")
    print(f"{'=' * 110}")

    real_df["size_bucket"] = pd.cut(real_df["n"],
        bins=[0, 5_000, 20_000, 50_000, 200_000, 1_000_000],
        labels=["<5K", "5K-20K", "20K-50K", "50K-200K", ">200K"])
    for bucket in ["<5K", "5K-20K", "20K-50K", "50K-200K", ">200K"]:
        subset = real_df[real_df["size_bucket"] == bucket]
        if len(subset) == 0:
            continue
        wins = subset["best_algorithm"].value_counts().to_dict()
        print(f"  {bucket:<12} ({len(subset):>3} arrays): {wins}")

    # ── Observations ──────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("KEY OBSERVATIONS FOR THESIS")
    print(f"{'=' * 110}")

    print("\n▸ Arrays where selection matters most (margin > 20%):")
    for _, row in real_df.iterrows():
        ts = {a: row[f"time_{a}"] for a in ALGORITHMS}
        best_t = min(ts.values())
        worst_t = max(ts.values())
        margin = (worst_t - best_t) / (best_t + 1e-15) * 100
        if margin > 20:
            winner = row["best_algorithm"]
            loser = max(ts, key=ts.get)
            print(f"  {row['array']:<65} → {winner} beats {loser} by {margin:.0f}%")

    # ── Save everything ───────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("SAVING RESULTS")
    print(f"{'=' * 110}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Drop the categorical bucket column for parquet save
    save_df = real_df.drop(columns=["size_bucket"], errors="ignore")

    parquet_path = OUTPUT_DIR / "f1_real_world_v2_results.parquet"
    save_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved {parquet_path.relative_to(PROJECT_ROOT)}  ({len(save_df)} rows)")

    csv_path = OUTPUT_DIR / "f1_real_world_v2_results.csv"
    save_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path.relative_to(PROJECT_ROOT)}  ({len(save_df)} rows)")

    config = {
        "version": "v2",
        "description": "Full-race F1 telemetry — all laps concatenated for large arrays",
        "sessions": [f"{y} {gp} {s}" for y, gp, s in SESSIONS],
        "n_arrays": len(save_df),
        "n_features": len(_FEATURE_NAMES),
        "feature_names": _FEATURE_NAMES,
        "algorithms": list(ALGORITHMS.keys()),
        "n_max": N_MAX,
        "array_size_range": [int(save_df["n"].min()), int(save_df["n"].max())],
        "winner_counts": winner_counts.to_dict(),
        "vbs_sbs_gap_pct": round(gap, 2),
        "sbs_algorithm": sbs_name,
        "channels": CHANNELS,
        "strategies": [
            "full_race_per_driver: all laps of one driver concatenated",
            "all_drivers_concat: all drivers speed data in one array",
        ],
        "v1_comparison": {
            "v1_sizes": "~700 (1 fastest lap)",
            "v2_sizes": f"{int(save_df['n'].min()):,}–{int(save_df['n'].max()):,} (all laps)",
        },
    }
    config_path = OUTPUT_DIR / "real_world_v2_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    print(f"  ✓ Saved {config_path.relative_to(PROJECT_ROOT)}")

    print(f"\n{'=' * 110}")
    print(f"DONE — v2 results saved to data/real_world_v2/")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
