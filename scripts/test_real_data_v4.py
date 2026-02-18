#!/usr/bin/env python3
"""
Real-World Validation v4 — Cross-Domain Combined Benchmark
=============================================================
Combines ALL existing data + NEW large-scale real sources into one mixed-workload
benchmark to measure the true VBS-SBS gap under realistic diversity.

Data sources:
  A. Existing (loaded from parquet):
     - Synthetic benchmark (720 arrays, 10K–2M)
     - F1 telemetry v2     (108 arrays, 34K–1.13M)
     - Financial + seismic  (149 arrays, 2K–309K)

  B. New real data (fetched live):
     - NOAA Climate: daily temps from 5+ stations, decades of data
     - NASA Near-Earth Objects: orbital parameters of ~40K asteroids
     - Open-Meteo historical weather: hourly data for major cities (100K+ per city)
     - Large synthetic-real hybrids: heavy-tailed, Zipfian, Brownian motion at 2M+ scale

Usage:
    source venv/bin/activate
    python scripts/test_real_data_v4.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Import our feature extraction (single source of truth) ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import extract_features, FEATURE_NAMES as _FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "all_samples.parquet"
V2_PATH = PROJECT_ROOT / "data" / "real_world_v2" / "f1_real_world_v2_results.parquet"
V3_PATH = PROJECT_ROOT / "data" / "real_world_v3" / "real_world_v3_results.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_world_v4"

N_MAX = 2_000_000
TIMING_REPEATS = 3
ALGOS = {
    "introsort": lambda a: np.sort(a, kind="quicksort"),
    "heapsort": lambda a: np.sort(a, kind="heapsort"),
    "timsort": lambda a: np.sort(a, kind="stable"),
}

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1: Load existing datasets
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_datasets() -> pd.DataFrame:
    """Load and unify all existing parquet results."""
    frames = []
    feat_cols = list(_FEATURE_NAMES)
    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    common_cols = ["array", "n", "domain"] + feat_cols + time_cols + ["best_algorithm"]

    # 1. Synthetic benchmark
    bench = pd.read_parquet(BENCHMARK_PATH)
    bench["array"] = bench["sample_id"]
    bench["domain"] = "synthetic"
    bench["best_algorithm"] = bench[time_cols].idxmin(axis=1).str.replace("time_", "")
    for c in common_cols:
        if c not in bench.columns:
            bench[c] = np.nan
    frames.append(bench[common_cols].copy())
    print(f"  Loaded synthetic benchmark: {len(bench)} rows")

    # 2. F1 v2
    f1v2 = pd.read_parquet(V2_PATH)
    f1v2["domain"] = "f1_telemetry"
    for c in common_cols:
        if c not in f1v2.columns:
            f1v2[c] = np.nan
    frames.append(f1v2[common_cols].copy())
    print(f"  Loaded F1 v2: {len(f1v2)} rows")

    # 3. Financial + seismic v3
    v3 = pd.read_parquet(V3_PATH)
    for c in common_cols:
        if c not in v3.columns:
            v3[c] = np.nan
    frames.append(v3[common_cols].copy())
    print(f"  Loaded v3 (finance+seismic): {len(v3)} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total existing: {len(combined)} rows")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2: Fetch new real data
# ═══════════════════════════════════════════════════════════════════════════

def fetch_open_meteo_hourly(lat: float, lon: float, city: str,
                             start: str = "1950-01-01",
                             end: str = "2025-12-31") -> list[tuple[str, np.ndarray]]:
    """Fetch hourly temperature + humidity + wind from Open-Meteo historical API."""
    arrays = []
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        for var in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]:
            vals = hourly.get(var)
            if vals:
                arr = np.array([v for v in vals if v is not None], dtype=np.float64)
                if len(arr) >= 1000:
                    arrays.append((f"weather_{city}_{var}", arr))
                    print(f"    weather_{city}_{var}: {len(arr):,} values")
    except Exception as e:
        print(f"    ⚠ {city} weather failed: {e}")
    return arrays


def fetch_nasa_neo() -> list[tuple[str, np.ndarray]]:
    """Fetch NASA Near-Earth Object parameters from SBDB API."""
    arrays = []
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    params = {
        "fields": "e,a,i,om,w,q,ad,per,H",
        "sb-class": "APO",      # Apollo asteroids
        "limit": "0",           # all
    }
    try:
        print("    Fetching NASA NEO database (Apollo asteroids)...")
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        fields = data.get("fields", [])
        print(f"    Got {len(rows)} asteroids, fields: {fields}")

        # Each field becomes an array
        field_names = {
            "e": "eccentricity", "a": "semi_major_axis", "i": "inclination",
            "om": "long_asc_node", "w": "arg_perihelion", "q": "perihelion_dist",
            "ad": "aphelion_dist", "per": "orbital_period", "H": "abs_magnitude",
        }
        for fi, fname in enumerate(fields):
            nice = field_names.get(fname, fname)
            vals = []
            for row in rows:
                try:
                    vals.append(float(row[fi]))
                except (ValueError, TypeError, IndexError):
                    pass
            if len(vals) >= 500:
                arr = np.array(vals, dtype=np.float64)
                arrays.append((f"neo_{nice}", arr))
                print(f"    neo_{nice}: {len(arr):,} values")
    except Exception as e:
        print(f"    ⚠ NASA NEO failed: {e}")
    return arrays


def fetch_usgs_earthquake_extended() -> list[tuple[str, np.ndarray]]:
    """Fetch a much larger earthquake dataset — 300K events."""
    arrays = []
    all_events = []
    # Fetch in yearly chunks to get more data
    year_ranges = [
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
    ]
    for start, end in year_ranges:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "csv",
            "starttime": start,
            "endtime": end,
            "minmagnitude": "0.5",
            "limit": "20000",
            "orderby": "time",
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            all_events.append(df)
            print(f"    Earthquakes {start[:4]}: {len(df):,} events")
        except Exception as e:
            print(f"    ⚠ Earthquakes {start[:4]}: {e}")

    if all_events:
        eq = pd.concat(all_events, ignore_index=True)
        print(f"    Total earthquakes: {len(eq):,}")

        for col, name in [
            ("mag", "eq_ext_magnitude"),
            ("depth", "eq_ext_depth"),
            ("latitude", "eq_ext_latitude"),
            ("longitude", "eq_ext_longitude"),
        ]:
            if col in eq.columns:
                vals = eq[col].dropna().values.astype(np.float64)
                if len(vals) >= 1000:
                    arrays.append((name, vals))
                    print(f"    {name}: {len(vals):,}")

        # Derived: inter-event time gaps
        if "time" in eq.columns:
            times = pd.to_datetime(eq["time"], errors="coerce").dropna()
            times = times.sort_values()
            gaps = np.diff(times.astype(np.int64) // 10**9).astype(np.float64)
            gaps = gaps[gaps > 0]
            if len(gaps) >= 1000:
                arrays.append(("eq_ext_time_gaps", gaps))
                print(f"    eq_ext_time_gaps: {len(gaps):,}")

        # Derived: magnitude × depth
        if "mag" in eq.columns and "depth" in eq.columns:
            valid = eq[["mag", "depth"]].dropna()
            cross = (valid["mag"] * valid["depth"]).values.astype(np.float64)
            if len(cross) >= 1000:
                arrays.append(("eq_ext_mag_x_depth", cross))

    return arrays


def generate_large_scale_arrays() -> list[tuple[str, np.ndarray]]:
    """Generate large-scale arrays with known statistical properties at 1M-5M scale."""
    arrays = []
    rng = np.random.default_rng(42)
    print("    Generating large-scale arrays...")

    # 1. Brownian motion (cumulative sum of random steps) — 2M
    steps = rng.standard_normal(2_000_000)
    bm = np.cumsum(steps)
    arrays.append(("largescale_brownian_2M", bm))
    print(f"    largescale_brownian_2M: {len(bm):,}")

    # 2. Heavy-tailed Pareto — 2M
    pareto = rng.pareto(a=1.5, size=2_000_000)
    arrays.append(("largescale_pareto_2M", pareto))
    print(f"    largescale_pareto_2M: {len(pareto):,}")

    # 3. Zipf/power-law (discrete) — 2M
    zipf = rng.zipf(a=1.5, size=2_000_000).astype(np.float64)
    arrays.append(("largescale_zipf_2M", zipf))
    print(f"    largescale_zipf_2M: {len(zipf):,}")

    # 4. Nearly sorted with noise — 2M
    nearly = np.arange(2_000_000, dtype=np.float64)
    noise_idx = rng.choice(2_000_000, size=40_000, replace=False)  # 2% swaps
    noise_pairs = noise_idx.reshape(-1, 2)
    for i, j in noise_pairs:
        nearly[i], nearly[j] = nearly[j], nearly[i]
    arrays.append(("largescale_nearly_sorted_2M", nearly))
    print(f"    largescale_nearly_sorted_2M: {len(nearly):,}")

    # 5. Reverse sorted — 2M
    rev = np.arange(2_000_000, 0, -1, dtype=np.float64)
    arrays.append(("largescale_reverse_sorted_2M", rev))
    print(f"    largescale_reverse_sorted_2M: {len(rev):,}")

    # 6. Exponential (long right tail) — 2M
    exp_arr = rng.exponential(scale=100.0, size=2_000_000)
    arrays.append(("largescale_exponential_2M", exp_arr))
    print(f"    largescale_exponential_2M: {len(exp_arr):,}")

    # 7. Bimodal (two gaussians) — 2M
    g1 = rng.normal(loc=-1000, scale=100, size=1_000_000)
    g2 = rng.normal(loc=1000, scale=100, size=1_000_000)
    bimodal = np.concatenate([g1, g2])
    rng.shuffle(bimodal)
    arrays.append(("largescale_bimodal_2M", bimodal))
    print(f"    largescale_bimodal_2M: {len(bimodal):,}")

    # 8. Sorted-runs concatenation (like merge-sort intermediate) — 2M
    chunks = []
    pos = 0
    while pos < 2_000_000:
        run_len = min(int(rng.integers(100, 5000)), 2_000_000 - pos)
        chunk = rng.standard_normal(run_len)
        chunk.sort()
        if rng.random() < 0.3:
            chunk = chunk[::-1]
        chunks.append(chunk)
        pos += run_len
    sorted_runs = np.concatenate(chunks)
    arrays.append(("largescale_sorted_runs_2M", sorted_runs))
    print(f"    largescale_sorted_runs_2M: {len(sorted_runs):,}")

    # 9. Student-t (heavy tails, covers the skewness/kurtosis gap) — 2M
    studentt = rng.standard_t(df=3, size=2_000_000)
    arrays.append(("largescale_student_t_2M", studentt))
    print(f"    largescale_student_t_2M: {len(studentt):,}")

    # 10. Few unique at scale — 2M (only 50 unique values)
    unique_vals = rng.uniform(-1e6, 1e6, size=50)
    few_uniq = rng.choice(unique_vals, size=2_000_000)
    arrays.append(("largescale_few_unique_2M", few_uniq))
    print(f"    largescale_few_unique_2M: {len(few_uniq):,}")

    return arrays


def fetch_all_new_data() -> list[tuple[str, np.ndarray]]:
    """Fetch all new real-world data sources."""
    all_arrays = []

    # ── Weather data from Open-Meteo (free, no API key) ──
    # Major cities with long historical records
    cities = [
        (48.8566, 2.3522, "paris", "1950-01-01", "2025-01-01"),
        (40.7128, -74.0060, "new_york", "1950-01-01", "2025-01-01"),
        (35.6762, 139.6503, "tokyo", "1950-01-01", "2025-01-01"),
        (51.5074, -0.1278, "london", "1950-01-01", "2025-01-01"),
        (-33.8688, 151.2093, "sydney", "1950-01-01", "2025-01-01"),
        (55.7558, 37.6173, "moscow", "1950-01-01", "2025-01-01"),
        (19.4326, -99.1332, "mexico_city", "1950-01-01", "2025-01-01"),
        (1.3521, 103.8198, "singapore", "1950-01-01", "2025-01-01"),
    ]

    print("\n  ── Fetching weather data (Open-Meteo, 8 cities × 4 variables) ──")
    for lat, lon, city, start, end in cities:
        arrs = fetch_open_meteo_hourly(lat, lon, city, start, end)
        all_arrays.extend(arrs)

    # ── NASA Near-Earth Objects ──
    print("\n  ── Fetching NASA Near-Earth Object data ──")
    arrs = fetch_nasa_neo()
    all_arrays.extend(arrs)

    # ── Extended earthquake data (5 years, 100K) ──
    print("\n  ── Fetching extended earthquake data (5 years) ──")
    arrs = fetch_usgs_earthquake_extended()
    all_arrays.extend(arrs)

    # ── Large-scale synthetic-real arrays (2M each) ──
    print("\n  ── Generating large-scale benchmark arrays (2M each) ──")
    arrs = generate_large_scale_arrays()
    all_arrays.extend(arrs)

    return all_arrays


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3: Feature extraction + timing on new data
# ═══════════════════════════════════════════════════════════════════════════

def process_new_arrays(new_arrays: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    """Extract features and time algorithms on new arrays."""
    results = []
    total = len(new_arrays)

    for idx, (desc, arr) in enumerate(new_arrays, 1):
        n = len(arr)
        print(f"  [{idx:3d}/{total}] {desc:50s} n={n:>12,} ... ", end="", flush=True)

        # Extract features
        f = extract_features(arr, n_max=N_MAX, sample_id=desc)

        # Time algorithms
        times = {}
        for algo_name, algo_fn in ALGOS.items():
            run_times = []
            for _ in range(TIMING_REPEATS):
                copy = arr.copy()
                t0 = time.perf_counter()
                algo_fn(copy)
                t1 = time.perf_counter()
                run_times.append(t1 - t0)
            times[f"time_{algo_name}"] = float(np.median(run_times))
        
        best = min(times, key=times.get).replace("time_", "")
        
        # Determine domain
        if desc.startswith("weather_"):
            domain = "weather"
        elif desc.startswith("neo_"):
            domain = "nasa_neo"
        elif desc.startswith("eq_ext_"):
            domain = "earthquake_ext"
        elif desc.startswith("largescale_"):
            domain = "largescale"
        else:
            domain = "other"

        row = {"array": desc, "n": n, "domain": domain}
        row.update(f)
        row.update(times)
        row["best_algorithm"] = best
        results.append(row)
        
        margin_pct = (max(times.values()) / min(times.values()) - 1) * 100
        print(f"→ {best:10s} (margin {margin_pct:5.1f}%)")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4: Combined analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_vbs_sbs(df: pd.DataFrame, label: str = "") -> dict:
    """Compute VBS-SBS gap for a dataframe."""
    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    vbs = df[time_cols].min(axis=1).sum()
    sbs_totals = {c.replace("time_", ""): df[c].sum() for c in time_cols}
    sbs_name = min(sbs_totals, key=sbs_totals.get)
    sbs = sbs_totals[sbs_name]
    gap = (sbs - vbs) / sbs * 100 if sbs > 0 else 0.0

    winners = df["best_algorithm"].value_counts().to_dict()

    return {
        "label": label,
        "arrays": len(df),
        "vbs_total": vbs,
        "sbs_total": sbs,
        "sbs_algo": sbs_name,
        "gap_pct": gap,
        "winners": winners,
    }


def print_section(title: str):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def main():
    print_section("CROSS-DOMAIN COMBINED BENCHMARK (v4)")
    print(f"  Feature extraction: feature_extraction.py (16 features)")
    print(f"  Algorithms: introsort, heapsort, timsort (all np.sort C-level)")
    print(f"  Timing: {TIMING_REPEATS} repeats, median")
    print(f"  N_MAX: {N_MAX:,}")

    # ── Step 1: Load existing data ──
    print_section("STEP 1: Loading existing datasets")
    existing = load_existing_datasets()

    # ── Step 2: Fetch new data ──
    print_section("STEP 2: Fetching new real-world data")
    new_arrays = fetch_all_new_data()
    print(f"\n  Total new arrays to process: {len(new_arrays)}")

    # ── Step 3: Process new arrays ──
    print_section("STEP 3: Feature extraction + timing on new data")
    new_df = process_new_arrays(new_arrays)
    print(f"\n  New data processed: {len(new_df)} arrays")
    print(f"  Size range: {new_df['n'].min():,} – {new_df['n'].max():,}")

    # ── Step 4: Combine everything ──
    print_section("STEP 4: Combining all datasets")
    combined = pd.concat([existing, new_df], ignore_index=True)
    print(f"  Total combined arrays: {len(combined)}")
    print(f"  Size range: {combined['n'].min():,} – {combined['n'].max():,}")
    print(f"\n  Arrays per domain:")
    for domain, count in combined["domain"].value_counts().items():
        print(f"    {domain:25s}: {count:5d}")

    # ── Step 5: Sanity checks on new data ──
    print_section("STEP 5: Sanity checks on new data")
    bounded = [f for f in _FEATURE_NAMES if f not in ("skewness_t", "kurtosis_excess_t")]
    issues = 0
    for f in bounded:
        if f in new_df.columns:
            lo, hi = new_df[f].min(), new_df[f].max()
            ok = lo >= -0.001 and hi <= 1.001
            status = "✓" if ok else "⚠ OUTSIDE"
            if not ok:
                issues += 1
            print(f"    {f:25s}: [{lo:.4f}, {hi:.4f}]  {status}")
    
    nan_count = new_df[list(_FEATURE_NAMES)].isna().sum().sum()
    inf_count = np.isinf(new_df[list(_FEATURE_NAMES)].select_dtypes(include=[np.number])).sum().sum()
    print(f"\n    NaN values: {nan_count}  {'✓' if nan_count == 0 else '⚠'}")
    print(f"    Inf values: {inf_count}  {'✓' if inf_count == 0 else '⚠'}")
    print(f"    Bounded feature issues: {issues}  {'✓ ALL PASSED' if issues == 0 else '⚠'}")

    # ── Step 6: Feature comparison (new vs benchmark) ──
    print_section("STEP 6: Feature range comparison (new data vs benchmark)")
    bench = existing[existing["domain"] == "synthetic"]
    print(f"\n  Benchmark: {len(bench)} samples | New data: {len(new_df)} arrays")
    print(f"\n  {'Feature':25s}  {'Bench min':>10s} {'Bench max':>10s} │ {'New min':>10s} {'New max':>10s} │ {'Status':>10s}")
    print("  " + "─" * 90)
    for f in _FEATURE_NAMES:
        if f in bench.columns and f in new_df.columns:
            bmin, bmax = bench[f].min(), bench[f].max()
            nmin, nmax = new_df[f].min(), new_df[f].max()
            outside = nmin < bmin - 0.01 or nmax > bmax + 0.01
            status = "⚠ OUTSIDE" if outside else "✓"
            print(f"  {f:25s}  {bmin:10.4f} {bmax:10.4f} │ {nmin:10.4f} {nmax:10.4f} │ {status:>10s}")

    # ── Step 7: VBS-SBS gap analysis ──
    print_section("STEP 7: VBS-SBS GAP ANALYSIS")
    
    # Per-domain gaps
    print("\n  Per-domain analysis:")
    print(f"  {'Domain':25s} {'Arrays':>7s} {'VBS (s)':>10s} {'SBS (s)':>10s} {'SBS algo':>10s} {'Gap':>8s}")
    print("  " + "─" * 75)
    
    domain_results = []
    for domain in sorted(combined["domain"].dropna().unique()):
        sub = combined[combined["domain"] == domain]
        r = compute_vbs_sbs(sub, domain)
        domain_results.append(r)
        print(f"  {domain:25s} {r['arrays']:7d} {r['vbs_total']:10.4f} {r['sbs_total']:10.4f} {r['sbs_algo']:>10s} {r['gap_pct']:7.1f}%")

    # Combined gap (THE KEY NUMBER)
    print()
    r_all = compute_vbs_sbs(combined, "ALL COMBINED")
    print(f"  {'*** ALL COMBINED ***':25s} {r_all['arrays']:7d} {r_all['vbs_total']:10.4f} {r_all['sbs_total']:10.4f} {r_all['sbs_algo']:>10s} {r_all['gap_pct']:7.1f}%")

    # Sub-combinations
    print("\n  Key sub-combinations:")
    
    # Real data only (no synthetic)
    real_only = combined[combined["domain"] != "synthetic"]
    r_real = compute_vbs_sbs(real_only, "Real data only")
    print(f"  {'Real data only':25s} {r_real['arrays']:7d} {r_real['vbs_total']:10.4f} {r_real['sbs_total']:10.4f} {r_real['sbs_algo']:>10s} {r_real['gap_pct']:7.1f}%")

    # New v4 data only
    r_new = compute_vbs_sbs(new_df, "New v4 data only")
    print(f"  {'New v4 data only':25s} {r_new['arrays']:7d} {r_new['vbs_total']:10.4f} {r_new['sbs_total']:10.4f} {r_new['sbs_algo']:>10s} {r_new['gap_pct']:7.1f}%")

    # Large arrays only (n > 100K)
    large = combined[combined["n"] > 100_000]
    if len(large) > 0:
        r_large = compute_vbs_sbs(large, "Large arrays (n>100K)")
        print(f"  {'Large (n>100K)':25s} {r_large['arrays']:7d} {r_large['vbs_total']:10.4f} {r_large['sbs_total']:10.4f} {r_large['sbs_algo']:>10s} {r_large['gap_pct']:7.1f}%")

    # ── Step 8: Win rates ──
    print_section("STEP 8: Overall win rates")
    winners = combined["best_algorithm"].value_counts()
    for algo, count in winners.items():
        pct = count / len(combined) * 100
        print(f"  {algo:15s}: {count:5d} wins ({pct:5.1f}%)")

    # By domain
    print("\n  Win rates per domain:")
    for domain in sorted(combined["domain"].dropna().unique()):
        sub = combined[combined["domain"] == domain]
        w = sub["best_algorithm"].value_counts().to_dict()
        print(f"  {domain:25s}: {w}")

    # ── Step 9: Where selection matters most ──
    print_section("STEP 9: Where selection matters most (margin > 50%)")
    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    combined["worst_time"] = combined[time_cols].max(axis=1)
    combined["best_time"] = combined[time_cols].min(axis=1)
    combined["margin_pct"] = (combined["worst_time"] / combined["best_time"] - 1) * 100
    
    big_margin = combined[combined["margin_pct"] > 50].sort_values("margin_pct", ascending=False)
    print(f"  Arrays with >50% margin: {len(big_margin)} / {len(combined)} ({len(big_margin)/len(combined)*100:.1f}%)")
    print(f"\n  Top 30 largest margins:")
    print(f"  {'Array':55s} {'n':>10s} {'Domain':>15s} {'Winner':>12s} {'Margin':>8s}")
    print("  " + "─" * 105)
    for _, row in big_margin.head(30).iterrows():
        arr_name = str(row.get("array", "?"))[:55]
        print(f"  {arr_name:55s} {int(row['n']):10,} {str(row.get('domain','')):>15s} {row['best_algorithm']:>12s} {row['margin_pct']:7.1f}%")

    # ── Step 10: Cross-test comparison table ──
    print_section("STEP 10: Cross-test comparison (v1 → v2 → v3 → v4)")
    print(f"""
  Test                                  Arrays       Size range  VBS-SBS gap  Domain
  ────────────────────────────────────────────────────────────────────────────────────────
  Synthetic benchmark                      720           10K–2M    18.8%      Synthetic
  Real-world v1 (F1 fastest lap)            35          ~700         5.1%      F1 telemetry
  Real-world v2 (F1 full race)             108   34K–1.13M          3.2%      F1 telemetry
  Real-world v3 (Finance + Seismic)        149    2K–309K           1.6%      Stock, crypto, earthquake
  Real-world v4 (Cross-domain)          {r_all['arrays']:>5d}   {int(combined['n'].min()):,}–{int(combined['n'].max()):,}      {r_all['gap_pct']:.1f}%      ALL DOMAINS COMBINED
  Real-world v4 (new data only)         {r_new['arrays']:>5d}   {int(new_df['n'].min()):,}–{int(new_df['n'].max()):,}      {r_new['gap_pct']:.1f}%      Weather, NASA, EQ, large-scale
""")

    # ── Step 11: Save everything ──
    print_section("SAVING RESULTS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save new v4 data separately
    new_df.to_parquet(OUTPUT_DIR / "real_world_v4_new_data.parquet", index=False)
    new_df.to_csv(OUTPUT_DIR / "real_world_v4_new_data.csv", index=False)
    print(f"  ✓ Saved new data: {OUTPUT_DIR / 'real_world_v4_new_data.parquet'} ({len(new_df)} rows)")

    # Save combined dataset
    combined.to_parquet(OUTPUT_DIR / "real_world_v4_combined.parquet", index=False)
    combined.to_csv(OUTPUT_DIR / "real_world_v4_combined.csv", index=False)
    print(f"  ✓ Saved combined: {OUTPUT_DIR / 'real_world_v4_combined.parquet'} ({len(combined)} rows)")

    # Save config
    config = {
        "test": "v4_cross_domain_combined",
        "date": "2026-02-18",
        "existing_sources": {
            "synthetic": str(BENCHMARK_PATH),
            "f1_v2": str(V2_PATH),
            "v3_finance_seismic": str(V3_PATH),
        },
        "new_sources": ["open_meteo_weather", "nasa_neo", "usgs_earthquake_ext", "large_scale_synthetic"],
        "n_max": N_MAX,
        "timing_repeats": TIMING_REPEATS,
        "algorithms": list(ALGOS.keys()),
        "feature_names": list(_FEATURE_NAMES),
        "total_arrays": len(combined),
        "new_arrays": len(new_df),
        "combined_gap_pct": round(r_all["gap_pct"], 2),
        "new_data_gap_pct": round(r_new["gap_pct"], 2),
    }
    with open(OUTPUT_DIR / "real_world_v4_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config: {OUTPUT_DIR / 'real_world_v4_config.json'}")

    print_section("DONE — v4 results saved to data/real_world_v4/")


if __name__ == "__main__":
    main()
