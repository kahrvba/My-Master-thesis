#!/usr/bin/env python3
"""
Real-World Feature Extraction Test — F1 Telemetry
===================================================
Fetches real F1 telemetry data, extracts 1D numeric arrays,
runs our feature extraction pipeline, and compares the feature
distributions against our synthetic benchmark data.

Purpose: Validate that features behave sensibly on real-world data
         before investing in model training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

# ── Import our feature extraction (single source of truth) ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import extract_features, FEATURE_NAMES as _FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "f1_cache"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "all_samples.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_world"
N_MAX = 2_000_000.0  # Same as benchmark pipeline


def fetch_session(year: int, gp: str, session_type: str) -> fastf1.core.Session:
    """Load an F1 session with telemetry."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def extract_arrays_from_session(session) -> list[tuple[str, np.ndarray]]:
    """
    Extract multiple 1D numeric arrays from an F1 session.
    Returns list of (description, array) tuples.
    """
    arrays = []

    for drv in session.drivers[:5]:  # Top 5 drivers
        try:
            laps = session.laps.pick_drivers(drv)
            driver_name = laps.iloc[0]["Driver"] if len(laps) > 0 else drv

            # Get fastest lap telemetry
            fastest = laps.pick_fastest()
            if fastest is None or fastest.empty if hasattr(fastest, 'empty') else fastest is None:
                continue

            tel = fastest.get_telemetry()
            if tel is None or len(tel) == 0:
                continue

            # Speed trace — the most natural sorting benchmark array
            if "Speed" in tel.columns:
                speed = tel["Speed"].dropna().to_numpy(dtype=np.float64)
                if len(speed) > 100:
                    arrays.append((f"{driver_name}_speed (n={len(speed)})", speed))

            # RPM — high-frequency oscillating signal
            if "RPM" in tel.columns:
                rpm = tel["RPM"].dropna().to_numpy(dtype=np.float64)
                if len(rpm) > 100:
                    arrays.append((f"{driver_name}_RPM (n={len(rpm)})", rpm))

            # Throttle — 0-100 range, many plateaus
            if "Throttle" in tel.columns:
                throttle = tel["Throttle"].dropna().to_numpy(dtype=np.float64)
                if len(throttle) > 100:
                    arrays.append((f"{driver_name}_throttle (n={len(throttle)})", throttle))

            # Brake — binary-ish (0 or 100), lots of duplicates
            if "Brake" in tel.columns:
                brake = tel["Brake"].dropna().to_numpy(dtype=np.float64)
                if len(brake) > 100:
                    arrays.append((f"{driver_name}_brake (n={len(brake)})", brake))

            # nGear — integer 0-8, very few unique values
            if "nGear" in tel.columns:
                gear = tel["nGear"].dropna().to_numpy(dtype=np.float64)
                if len(gear) > 100:
                    arrays.append((f"{driver_name}_gear (n={len(gear)})", gear))

            # DRS — 0/1 binary signal
            if "DRS" in tel.columns:
                drs = tel["DRS"].dropna().to_numpy(dtype=np.float64)
                if len(drs) > 100:
                    arrays.append((f"{driver_name}_DRS (n={len(drs)})", drs))

            # Distance — monotonically increasing (sorted by definition)
            if "Distance" in tel.columns:
                dist = tel["Distance"].dropna().to_numpy(dtype=np.float64)
                if len(dist) > 100:
                    arrays.append((f"{driver_name}_distance (n={len(dist)})", dist))

        except Exception as e:
            print(f"  ⚠ Skipping driver {drv}: {e}")
            continue

    return arrays


def print_feature_comparison(real_features: pd.DataFrame, benchmark_path: Path):
    """Compare real-world feature ranges against synthetic benchmark."""
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON: Real F1 Data vs Synthetic Benchmark")
    print("=" * 80)

    if benchmark_path.exists():
        bench = pd.read_parquet(benchmark_path)[_FEATURE_NAMES]
        print(f"\nBenchmark: {len(bench)} samples | Real: {len(real_features)} arrays\n")
        print(f"{'Feature':<24} {'Bench min':>10} {'Bench max':>10} │ {'Real min':>10} {'Real max':>10} │ {'In range?':>10}")
        print("─" * 100)
        for feat in _FEATURE_NAMES:
            b_min = bench[feat].min()
            b_max = bench[feat].max()
            r_min = real_features[feat].min()
            r_max = real_features[feat].max()
            in_range = "✓" if r_min >= b_min - 0.1 and r_max <= b_max + 0.1 else "⚠ OUTSIDE"
            print(f"{feat:<24} {b_min:>10.4f} {b_max:>10.4f} │ {r_min:>10.4f} {r_max:>10.4f} │ {in_range:>10}")
    else:
        print("  ⚠ Benchmark parquet not found — showing real data stats only")
        print(real_features[_FEATURE_NAMES].describe().T.to_string())


def main():
    print("=" * 80)
    print("REAL-WORLD FEATURE EXTRACTION TEST — F1 TELEMETRY")
    print("=" * 80)

    # ── Fetch F1 data ─────────────────────────────────────────────────────
    print("\n▸ Fetching F1 2024 Bahrain GP Race telemetry...")
    try:
        session = fetch_session(2024, "Bahrain", "R")
    except Exception as e:
        print(f"  ✗ Failed to fetch session: {e}")
        print("  Trying 2023 instead...")
        try:
            session = fetch_session(2023, "Bahrain", "R")
        except Exception as e2:
            print(f"  ✗ Also failed: {e2}")
            sys.exit(1)

    # ── Extract arrays ────────────────────────────────────────────────────
    print("\n▸ Extracting telemetry arrays...")
    arrays = extract_arrays_from_session(session)
    print(f"  Found {len(arrays)} arrays from {session.event['EventName']}")

    if not arrays:
        print("  ✗ No arrays extracted — check telemetry availability")
        sys.exit(1)

    # ── Run feature extraction + timing ─────────────────────────────────
    print("\n▸ Running feature extraction + algorithm timing on real data...")
    print(f"{'─' * 100}")
    print(f"{'Array':<40} {'n':>8} │ {'adj_sort':>8} {'dup_rat':>8} {'inv_rat':>8} {'entropy':>8} {'runs':>8} │ {'winner':>10}")
    print(f"{'─' * 100}")

    import time

    ALGORITHMS = {
        "introsort": lambda a: np.sort(a, kind="quicksort"),
        "heapsort": lambda a: np.sort(a, kind="heapsort"),
        "timsort": lambda a: np.sort(a, kind="stable"),
    }

    rows = []
    for desc, arr in arrays:
        f = extract_features(arr, n_max=N_MAX, sample_id=desc)

        # Time each algorithm (3 repeats, take median)
        timings = {}
        for algo_name, algo_fn in ALGORITHMS.items():
            times_list = []
            for _ in range(3):
                copy = arr.copy()
                t0 = time.perf_counter()
                algo_fn(copy)
                t1 = time.perf_counter()
                times_list.append(t1 - t0)
            timings[f"time_{algo_name}"] = float(np.median(times_list))

        winner = min(timings, key=timings.get).replace("time_", "")
        row = {"array": desc, "n": len(arr), **f, **timings, "best_algorithm": winner}
        rows.append(row)

        print(f"{desc:<40} {len(arr):>8} │ "
              f"{f['adj_sorted_ratio']:>8.4f} "
              f"{f['duplicate_ratio']:>8.4f} "
              f"{f['inversion_ratio']:>8.4f} "
              f"{f['entropy_ratio']:>8.4f} "
              f"{f['runs_ratio']:>8.4f} │ "
              f"{winner:>10}")

    real_df = pd.DataFrame(rows)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SANITY CHECKS")
    print(f"{'=' * 80}")

    issues = 0

    # Check: all features finite
    feat_vals = real_df[_FEATURE_NAMES].to_numpy()
    nan_count = np.sum(np.isnan(feat_vals))
    inf_count = np.sum(np.isinf(feat_vals))
    if nan_count > 0 or inf_count > 0:
        print(f"  ✗ NaN: {nan_count}, Inf: {inf_count}")
        issues += 1
    else:
        print(f"  ✓ No NaN or Inf in any feature")

    # Check: bounded features in [0, 1]
    bounded = [f for f in _FEATURE_NAMES if f not in ("skewness_t", "kurtosis_excess_t")]
    for feat in bounded:
        mn, mx = real_df[feat].min(), real_df[feat].max()
        if mn < -1e-9 or mx > 1.0 + 1e-9:
            print(f"  ✗ {feat} out of bounds: [{mn:.6f}, {mx:.6f}]")
            issues += 1

    if issues == 0:
        print(f"  ✓ All bounded features in [0, 1]")

    # Check: distance arrays should have adj_sorted_ratio ≈ 1.0
    for _, row in real_df.iterrows():
        if "distance" in row["array"]:
            asr = row["adj_sorted_ratio"]
            if asr > 0.95:
                print(f"  ✓ Distance array correctly detected as sorted (adj_sorted={asr:.4f})")
            else:
                print(f"  ✗ Distance array NOT detected as sorted (adj_sorted={asr:.4f})")
                issues += 1

    # Check: gear arrays should have high duplicate_ratio
    for _, row in real_df.iterrows():
        if "gear" in row["array"]:
            dr = row["duplicate_ratio"]
            if dr > 0.95:
                print(f"  ✓ Gear array correctly detected as few-unique (dup_ratio={dr:.4f})")
            else:
                print(f"  ✗ Gear array NOT detected as few-unique (dup_ratio={dr:.4f})")
                issues += 1

    # Check: brake arrays should have high duplicate_ratio (mostly 0 or 100)
    for _, row in real_df.iterrows():
        if "brake" in row["array"]:
            dr = row["duplicate_ratio"]
            if dr > 0.5:
                print(f"  ✓ Brake array has expected high duplicates (dup_ratio={dr:.4f})")

    # ── Compare with benchmark ────────────────────────────────────────────
    print_feature_comparison(real_df, BENCHMARK_PATH)

    # ── Interesting observations ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("OBSERVATIONS FOR THESIS")
    print(f"{'=' * 80}")

    # Which arrays would benefit most from smart algorithm selection?
    print("\n▸ Arrays with extreme feature values (interesting for selection):")
    for _, row in real_df.iterrows():
        notes = []
        if row["adj_sorted_ratio"] > 0.95:
            notes.append("nearly sorted → Timsort wins")
        if row["adj_sorted_ratio"] < 0.05:
            notes.append("reverse sorted → Timsort wins")
        if row["duplicate_ratio"] > 0.95:
            notes.append("many duplicates")
        if row["inversion_ratio"] < 0.05:
            notes.append("low inversions → Timsort territory")
        if row["entropy_ratio"] < 0.2:
            notes.append("low entropy")
        if notes:
            print(f"  {row['array']:<40} → {', '.join(notes)}")

    # Summary
    print(f"\n{'=' * 80}")
    if issues == 0:
        print("✓ ALL CHECKS PASSED — Features work correctly on real F1 data")
    else:
        print(f"⚠ {issues} ISSUES — Review above")
    print(f"{'=' * 80}")

    # ── Algorithm timing summary ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("ALGORITHM TIMING SUMMARY (Real F1 Data)")
    print(f"{'=' * 80}")

    winner_counts = real_df["best_algorithm"].value_counts()
    total = len(real_df)
    print(f"\n  Win counts ({total} arrays):")
    for algo, count in winner_counts.items():
        print(f"    {algo:<12} {count:>3} wins  ({100*count/total:.1f}%)")

    # VBS vs SBS on real data
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

    # Per-array timing table
    print(f"\n{'─' * 100}")
    print(f"{'Array':<40} {'introsort':>12} {'heapsort':>12} {'timsort':>12} │ {'winner':>10} {'margin':>8}")
    print(f"{'─' * 100}")
    for _, row in real_df.iterrows():
        ts = {a: row[f"time_{a}"] for a in ["introsort", "heapsort", "timsort"]}
        best_t = min(ts.values())
        second_t = sorted(ts.values())[1]
        margin = (second_t - best_t) / best_t * 100
        print(f"{row['array']:<40} "
              f"{ts['introsort']:>12.6f} {ts['heapsort']:>12.6f} {ts['timsort']:>12.6f} │ "
              f"{row['best_algorithm']:>10} {margin:>7.1f}%")

    # ── Save everything ───────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full results as parquet
    parquet_path = OUTPUT_DIR / "f1_real_world_results.parquet"
    real_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved {parquet_path.relative_to(PROJECT_ROOT)}  ({len(real_df)} rows)")

    # Save full results as CSV for easy viewing
    csv_path = OUTPUT_DIR / "f1_real_world_results.csv"
    real_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path.relative_to(PROJECT_ROOT)}  ({len(real_df)} rows)")

    # Save config / metadata
    import json
    config = {
        "source": "F1 Telemetry",
        "session": str(session.event["EventName"]),
        "year": int(session.event.year) if hasattr(session.event, "year") else 2024,
        "session_type": "Race",
        "n_arrays": len(real_df),
        "n_features": len(_FEATURE_NAMES),
        "feature_names": _FEATURE_NAMES,
        "algorithms": list(ALGORITHMS.keys()),
        "n_max": N_MAX,
        "winner_counts": winner_counts.to_dict(),
        "vbs_sbs_gap_pct": round(gap, 2),
        "sbs_algorithm": sbs_name,
        "telemetry_channels": ["Speed", "RPM", "Throttle", "Brake", "nGear", "DRS", "Distance"],
    }
    config_path = OUTPUT_DIR / "real_world_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  ✓ Saved {config_path.relative_to(PROJECT_ROOT)}")

    print(f"\n{'=' * 80}")
    print(f"DONE — All results saved to data/real_world/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
