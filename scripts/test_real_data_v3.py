#!/usr/bin/env python3
"""
Real-World Feature Extraction Test v3 — Financial Markets + Seismic Data
=========================================================================
v1: F1 telemetry, 1 fastest lap, ~700 samples          → timsort dominated
v2: F1 telemetry, all laps concatenated, 34K–1.13M     → 3-way competition
v3: Completely different domains — stock markets + earthquakes
    → proves feature extraction generalises beyond motorsport sensor data

Data Sources:
  1. Stock Market (yfinance) — 20 major stocks, daily data, 20+ years
     Arrays: Close prices, Volumes, Returns, High-Low ranges
  2. Cryptocurrency (yfinance) — BTC, ETH, SOL, etc.
     Arrays: Prices, volumes, returns (extreme volatility)
  3. Earthquake Catalog (USGS API) — global seismic events, 2+ years
     Arrays: Magnitudes, depths, coordinates (Gutenberg-Richter distribution)

Why these domains:
  - Financial: random walk, heavy tails, clustered volatility
  - Seismic: power-law, bimodal depth, extreme skew
  - Both are fundamentally different from F1 telemetry (periodic sensor signals)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import our feature extraction (single source of truth) ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import extract_features, FEATURE_NAMES as _FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "all_samples.parquet"
V1_PATH = PROJECT_ROOT / "data" / "real_world" / "f1_real_world_results.parquet"
V2_PATH = PROJECT_ROOT / "data" / "real_world_v2" / "f1_real_world_v2_results.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_world_v3"
N_MAX = 2_000_000.0

ALGORITHMS = {
    "introsort": lambda a: np.sort(a, kind="quicksort"),
    "heapsort":  lambda a: np.sort(a, kind="heapsort"),
    "timsort":   lambda a: np.sort(a, kind="stable"),
}

# ── Stock tickers ─────────────────────────────────────────────────────────
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM",
    "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", "MA", "BAC", "DIS",
    "NFLX", "INTC", "AMD", "CRM", "ORCL", "CSCO", "ADBE", "PFE",
    "KO", "PEP", "MRK", "ABBV",
]

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"]


def fetch_stock_data() -> list[tuple[str, np.ndarray]]:
    """Fetch daily stock data for 30 major stocks, 20+ years."""
    import yfinance as yf

    arrays = []
    print(f"\n  Downloading {len(STOCK_TICKERS)} stocks...")

    # Download all at once (faster)
    data = yf.download(
        STOCK_TICKERS,
        period="max",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # ── Per-stock arrays ──────────────────────────────────────────────
    all_close = []
    all_volume = []
    all_returns = []
    all_hl_range = []

    for ticker in STOCK_TICKERS:
        try:
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                tk = data[ticker].dropna(how="all")
            else:
                tk = data.dropna(how="all")

            if len(tk) < 1000:
                print(f"    ⚠ {ticker}: only {len(tk)} rows, skipping")
                continue

            close = tk["Close"].dropna().to_numpy(dtype=np.float64)
            volume = tk["Volume"].dropna().to_numpy(dtype=np.float64)
            high = tk["High"].dropna().to_numpy(dtype=np.float64)
            low = tk["Low"].dropna().to_numpy(dtype=np.float64)

            if len(close) > 1000:
                # Per-stock arrays
                arrays.append((f"stock_{ticker}_close (n={len(close)})", close))
                arrays.append((f"stock_{ticker}_volume (n={len(volume)})", volume))

                # Daily returns
                returns = np.diff(np.log(close + 1e-12))
                if len(returns) > 1000:
                    arrays.append((f"stock_{ticker}_log_returns (n={len(returns)})", returns))

                # High-Low range (intraday volatility proxy)
                hl = high - low
                if len(hl) > 1000:
                    arrays.append((f"stock_{ticker}_hl_range (n={len(hl)})", hl))

                # Collect for mega-concat
                all_close.append(close)
                all_volume.append(volume)
                all_returns.append(returns)
                all_hl_range.append(hl)

                print(f"    ✓ {ticker}: {len(close)} days")

        except Exception as e:
            print(f"    ⚠ {ticker}: {e}")
            continue

    # ── Mega-concat arrays (all stocks combined) ──────────────────────
    if all_close:
        mega_close = np.concatenate(all_close)
        arrays.append((f"stock_ALL_close (n={len(mega_close)})", mega_close))
    if all_volume:
        mega_vol = np.concatenate(all_volume)
        arrays.append((f"stock_ALL_volume (n={len(mega_vol)})", mega_vol))
    if all_returns:
        mega_ret = np.concatenate(all_returns)
        arrays.append((f"stock_ALL_log_returns (n={len(mega_ret)})", mega_ret))
    if all_hl_range:
        mega_hl = np.concatenate(all_hl_range)
        arrays.append((f"stock_ALL_hl_range (n={len(mega_hl)})", mega_hl))

    print(f"  ✓ Stock data: {len(arrays)} arrays total")
    return arrays


def fetch_crypto_data() -> list[tuple[str, np.ndarray]]:
    """Fetch daily crypto data (high volatility, fat tails)."""
    import yfinance as yf

    arrays = []
    print(f"\n  Downloading {len(CRYPTO_TICKERS)} cryptocurrencies...")

    all_close = []
    all_volume = []
    all_returns = []

    for ticker in CRYPTO_TICKERS:
        try:
            tk = yf.download(ticker, period="max", interval="1d", progress=False)
            if len(tk) < 500:
                print(f"    ⚠ {ticker}: only {len(tk)} rows, skipping")
                continue

            close = tk["Close"].dropna().to_numpy(dtype=np.float64).ravel()
            volume = tk["Volume"].dropna().to_numpy(dtype=np.float64).ravel()

            arrays.append((f"crypto_{ticker}_close (n={len(close)})", close))
            arrays.append((f"crypto_{ticker}_volume (n={len(volume)})", volume))

            returns = np.diff(np.log(close + 1e-12))
            if len(returns) > 500:
                arrays.append((f"crypto_{ticker}_log_returns (n={len(returns)})", returns))

            all_close.append(close)
            all_volume.append(volume)
            all_returns.append(returns)

            print(f"    ✓ {ticker}: {len(close)} days")

        except Exception as e:
            print(f"    ⚠ {ticker}: {e}")
            continue

    # Mega-concat
    if all_close:
        mega = np.concatenate(all_close)
        arrays.append((f"crypto_ALL_close (n={len(mega)})", mega))
    if all_volume:
        mega = np.concatenate(all_volume)
        arrays.append((f"crypto_ALL_volume (n={len(mega)})", mega))
    if all_returns:
        mega = np.concatenate(all_returns)
        arrays.append((f"crypto_ALL_log_returns (n={len(mega)})", mega))

    print(f"  ✓ Crypto data: {len(arrays)} arrays total")
    return arrays


def fetch_earthquake_data() -> list[tuple[str, np.ndarray]]:
    """Fetch earthquake catalog from USGS API (no auth needed)."""
    arrays = []
    print(f"\n  Fetching earthquake data from USGS API...")

    # Fetch in chunks (USGS limits to 20K per query)
    all_data = []
    date_ranges = [
        ("2023-01-01", "2023-06-30"),
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
    ]

    for start, end in date_ranges:
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=csv&starttime={start}&endtime={end}"
            f"&minmagnitude=0.5&orderby=time&limit=20000"
        )
        try:
            print(f"    Fetching {start} to {end}...")
            req = urllib.request.Request(url, headers={"User-Agent": "MasterThesis/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                csv_text = resp.read().decode("utf-8")
            chunk = pd.read_csv(io.StringIO(csv_text))
            all_data.append(chunk)
            print(f"    ✓ Got {len(chunk)} events")
        except Exception as e:
            print(f"    ⚠ Failed {start}-{end}: {e}")
            continue

    if not all_data:
        print("  ✗ No earthquake data fetched")
        return arrays

    eq = pd.concat(all_data, ignore_index=True)
    print(f"  Total earthquake events: {len(eq)}")

    # ── Extract arrays from earthquake catalog ────────────────────────
    channels = {
        "magnitude": "mag",
        "depth_km": "depth",
        "latitude": "latitude",
        "longitude": "longitude",
    }

    for label, col in channels.items():
        if col in eq.columns:
            vals = eq[col].dropna().to_numpy(dtype=np.float64)
            if len(vals) > 1000:
                arrays.append((f"earthquake_{label} (n={len(vals)})", vals))

    # Derived: distance from origin (0,0)
    if "latitude" in eq.columns and "longitude" in eq.columns:
        lat = eq["latitude"].dropna().to_numpy(dtype=np.float64)
        lon = eq["longitude"].dropna().to_numpy(dtype=np.float64)
        mn = min(len(lat), len(lon))
        dist = np.sqrt(lat[:mn]**2 + lon[:mn]**2)
        if len(dist) > 1000:
            arrays.append((f"earthquake_dist_from_origin (n={len(dist)})", dist))

    # Derived: time gaps between events (seconds)
    if "time" in eq.columns:
        try:
            times = pd.to_datetime(eq["time"]).sort_values()
            gaps = times.diff().dt.total_seconds().dropna().to_numpy(dtype=np.float64)
            gaps = gaps[gaps > 0]  # Remove zero/negative gaps
            if len(gaps) > 1000:
                arrays.append((f"earthquake_time_gaps_sec (n={len(gaps)})", gaps))
        except Exception:
            pass

    # Derived: magnitude * depth interaction
    if "mag" in eq.columns and "depth" in eq.columns:
        mag = eq["mag"].dropna().to_numpy(dtype=np.float64)
        dep = eq["depth"].dropna().to_numpy(dtype=np.float64)
        mn = min(len(mag), len(dep))
        interaction = mag[:mn] * dep[:mn]
        if len(interaction) > 1000:
            arrays.append((f"earthquake_mag_x_depth (n={len(interaction)})", interaction))

    print(f"  ✓ Earthquake data: {len(arrays)} arrays total")
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
    print("=" * 120)
    print("REAL-WORLD FEATURE EXTRACTION TEST v3 — FINANCIAL MARKETS + SEISMIC DATA")
    print("=" * 120)
    print(f"\nDomain shift from F1 telemetry: financial random walks, heavy tails,")
    print(f"seismic power-law distributions, earthquake inter-arrival times.\n")

    all_arrays: list[tuple[str, np.ndarray]] = []

    # ── Fetch from 3 sources ──────────────────────────────────────────────
    print("─" * 120)
    print("SOURCE 1: STOCK MARKET (30 major US stocks, daily, max history)")
    print("─" * 120)
    stock_arrays = fetch_stock_data()
    all_arrays.extend(stock_arrays)

    print("\n" + "─" * 120)
    print("SOURCE 2: CRYPTOCURRENCY (5 coins, daily, max history)")
    print("─" * 120)
    crypto_arrays = fetch_crypto_data()
    all_arrays.extend(crypto_arrays)

    print("\n" + "─" * 120)
    print("SOURCE 3: EARTHQUAKE CATALOG (USGS, 2023–2025, global)")
    print("─" * 120)
    eq_arrays = fetch_earthquake_data()
    all_arrays.extend(eq_arrays)

    if not all_arrays:
        print("\n  ✗ No arrays extracted — exiting")
        sys.exit(1)

    # Sort by size descending
    all_arrays.sort(key=lambda x: len(x[1]), reverse=True)

    # ── Size summary ──────────────────────────────────────────────────────
    sizes = [len(a) for _, a in all_arrays]
    print(f"\n{'=' * 120}")
    print(f"ARRAY SIZE SUMMARY: {len(all_arrays)} arrays from 3 domains")
    print(f"  Min: {min(sizes):,}  |  Median: {int(np.median(sizes)):,}  |  Max: {max(sizes):,}")
    print(f"  Stock arrays: {len(stock_arrays)}  |  Crypto arrays: {len(crypto_arrays)}  |  Earthquake arrays: {len(eq_arrays)}")
    print(f"  Arrays > 10K: {sum(1 for s in sizes if s > 10_000)}  |  > 50K: {sum(1 for s in sizes if s > 50_000)}  |  > 100K: {sum(1 for s in sizes if s > 100_000)}")
    print(f"{'=' * 120}")

    # ── Feature extraction + timing ───────────────────────────────────────
    print(f"\n▸ Running feature extraction + algorithm timing...")
    print(f"{'─' * 150}")
    print(f"{'Array':<55} {'n':>8} │ {'adj_sort':>8} {'dup_rat':>8} {'inv_rat':>8} {'entropy':>8} {'skew_t':>8} {'kurt_t':>8} │ {'winner':>10} {'t_win':>10}")
    print(f"{'─' * 150}")

    rows = []
    for desc, arr in all_arrays:
        f = extract_features(arr, n_max=N_MAX, sample_id=desc)
        timings = time_algorithms(arr)

        winner = min(timings, key=timings.get).replace("time_", "")
        best_time = min(timings.values())
        row = {"array": desc, "n": len(arr), **f, **timings, "best_algorithm": winner}

        # Tag source domain
        if desc.startswith("stock_"):
            row["domain"] = "stock"
        elif desc.startswith("crypto_"):
            row["domain"] = "crypto"
        elif desc.startswith("earthquake_"):
            row["domain"] = "earthquake"
        else:
            row["domain"] = "unknown"

        rows.append(row)

        print(f"{desc:<55} {len(arr):>8,} │ "
              f"{f['adj_sorted_ratio']:>8.4f} "
              f"{f['duplicate_ratio']:>8.4f} "
              f"{f['inversion_ratio']:>8.4f} "
              f"{f['entropy_ratio']:>8.4f} "
              f"{f['skewness_t']:>8.4f} "
              f"{f['kurtosis_excess_t']:>8.4f} │ "
              f"{winner:>10} {best_time:>10.6f}")

    real_df = pd.DataFrame(rows)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("SANITY CHECKS")
    print(f"{'=' * 120}")

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

    # Stock close prices should be mostly increasing over 20 years (bull market)
    for _, row in real_df.iterrows():
        if "_close" in row["array"] and "ALL" not in row["array"] and "stock_" in row["array"]:
            asr = row["adj_sorted_ratio"]
            if asr > 0.45:  # More up days than down → positive drift
                print(f"  ✓ {row['array'][:40]}: adj_sorted={asr:.4f} (positive drift expected)")
            break  # Just check one

    # Earthquake magnitudes should be heavily right-skewed (Gutenberg-Richter)
    for _, row in real_df.iterrows():
        if "earthquake_magnitude" in row["array"]:
            sk = row["skewness_t"]
            print(f"  {'✓' if sk > 0 else '✗'} Earthquake magnitudes: skewness_t={sk:.4f} (right-skewed expected)")
            break

    # Log returns should be near-symmetric (skewness ≈ 0)
    for _, row in real_df.iterrows():
        if "log_returns" in row["array"] and "ALL" in row["array"] and "stock_" in row["array"]:
            sk = row["skewness_t"]
            print(f"  {'✓' if abs(sk) < 1.5 else '✗'} Stock log returns: skewness_t={sk:.4f} (near-symmetric expected)")
            break

    if issues == 0:
        print(f"\n  ✓ ALL SANITY CHECKS PASSED")
    else:
        print(f"\n  ⚠ {issues} issues found")

    # ── Feature comparison vs benchmark ───────────────────────────────────
    print(f"\n{'=' * 120}")
    print("FEATURE COMPARISON: v3 (Financial + Seismic) vs Synthetic Benchmark")
    print(f"{'=' * 120}")

    if BENCHMARK_PATH.exists():
        bench = pd.read_parquet(BENCHMARK_PATH)[_FEATURE_NAMES]
        print(f"\nBenchmark: {len(bench)} samples | Real v3: {len(real_df)} arrays\n")
        print(f"{'Feature':<24} {'Bench min':>10} {'Bench max':>10} │ {'Real min':>10} {'Real max':>10} │ {'Status':>10}")
        print("─" * 100)
        for feat in _FEATURE_NAMES:
            b_min, b_max = float(bench[feat].min()), float(bench[feat].max())
            r_min, r_max = float(real_df[feat].min()), float(real_df[feat].max())
            in_range = "✓" if r_min >= b_min - 0.1 and r_max <= b_max + 0.1 else "⚠ OUTSIDE"
            print(f"{feat:<24} {b_min:>10.4f} {b_max:>10.4f} │ {r_min:>10.4f} {r_max:>10.4f} │ {in_range:>10}")

    # ── Feature comparison vs F1 (v2) ─────────────────────────────────────
    if V2_PATH.exists():
        print(f"\n{'=' * 120}")
        print("FEATURE COMPARISON: v3 (Financial + Seismic) vs v2 (F1 Telemetry)")
        print(f"{'=' * 120}")
        v2 = pd.read_parquet(V2_PATH)
        print(f"\nF1 v2: {len(v2)} arrays | v3: {len(real_df)} arrays\n")
        print(f"{'Feature':<24} {'F1 mean':>10} {'F1 std':>10} │ {'v3 mean':>10} {'v3 std':>10} │ {'Shift?':>10}")
        print("─" * 100)
        for feat in _FEATURE_NAMES:
            f1_mean = float(v2[feat].mean())
            f1_std = float(v2[feat].std())
            v3_mean = float(real_df[feat].mean())
            v3_std = float(real_df[feat].std())
            # Cohen's d for shift magnitude
            pooled_std = np.sqrt((f1_std**2 + v3_std**2) / 2) + 1e-12
            d = abs(f1_mean - v3_mean) / pooled_std
            shift = "LARGE" if d > 0.8 else ("MEDIUM" if d > 0.5 else "small")
            print(f"{feat:<24} {f1_mean:>10.4f} {f1_std:>10.4f} │ {v3_mean:>10.4f} {v3_std:>10.4f} │ {shift:>10}")

    # ── Algorithm timing summary ──────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("ALGORITHM TIMING SUMMARY (v3 — Financial + Seismic)")
    print(f"{'=' * 120}")

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

    # ── Per-domain breakdown ──────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("WIN RATES BY DOMAIN")
    print(f"{'=' * 120}")

    for domain in ["stock", "crypto", "earthquake"]:
        subset = real_df[real_df["domain"] == domain]
        if len(subset) == 0:
            continue
        wins = subset["best_algorithm"].value_counts().to_dict()
        dm = subset[time_cols].to_numpy()
        d_vbs = dm.min(axis=1).sum()
        d_sbs_idx = dm.sum(axis=0).argmin()
        d_sbs = dm[:, d_sbs_idx].sum()
        d_gap = (d_sbs - d_vbs) / d_vbs * 100
        d_sbs_name = time_cols[d_sbs_idx].replace("time_", "")
        print(f"\n  {domain.upper()} ({len(subset)} arrays):")
        print(f"    Winners: {wins}")
        print(f"    VBS-SBS gap: {d_gap:.1f}% (SBS={d_sbs_name})")

    # ── Per-array timing detail ───────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("PER-ARRAY TIMING DETAIL")
    print(f"{'=' * 120}")
    print(f"\n{'Array':<55} {'introsort':>12} {'heapsort':>12} {'timsort':>12} │ {'winner':>10} {'margin':>8}")
    print(f"{'─' * 120}")
    for _, row in real_df.iterrows():
        ts = {a: row[f"time_{a}"] for a in ALGORITHMS}
        best_t = min(ts.values())
        second_t = sorted(ts.values())[1]
        margin = (second_t - best_t) / (best_t + 1e-15) * 100
        print(f"{row['array']:<55} "
              f"{ts['introsort']:>12.6f} {ts['heapsort']:>12.6f} {ts['timsort']:>12.6f} │ "
              f"{row['best_algorithm']:>10} {margin:>7.1f}%")

    # ── Win rates by array type ───────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("WIN RATES BY ARRAY TYPE")
    print(f"{'=' * 120}")

    # Categorize by data type
    type_map = {
        "close": "Price (close)",
        "volume": "Volume",
        "log_returns": "Log returns",
        "hl_range": "High-Low range",
        "magnitude": "Eq. magnitude",
        "depth": "Eq. depth",
        "latitude": "Eq. latitude",
        "longitude": "Eq. longitude",
        "dist_from_origin": "Eq. distance",
        "time_gaps": "Eq. time gaps",
        "mag_x_depth": "Eq. mag×depth",
    }

    for key, label in type_map.items():
        subset = real_df[real_df["array"].str.contains(key, case=False)]
        if len(subset) == 0:
            continue
        wins = subset["best_algorithm"].value_counts().to_dict()
        avg_n = int(subset["n"].mean())
        print(f"  {label:<20} ({len(subset):>2} arrays, avg n={avg_n:>8,}): {wins}")

    # ── Key observations ──────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("KEY OBSERVATIONS FOR THESIS")
    print(f"{'=' * 120}")

    print("\n▸ Arrays where selection matters most (best vs worst margin > 30%):")
    for _, row in real_df.iterrows():
        ts = {a: row[f"time_{a}"] for a in ALGORITHMS}
        best_t = min(ts.values())
        worst_t = max(ts.values())
        margin = (worst_t - best_t) / (best_t + 1e-15) * 100
        if margin > 30:
            winner = row["best_algorithm"]
            loser = max(ts, key=ts.get)
            print(f"  {row['array']:<55} → {winner} beats {loser} by {margin:.0f}%")

    # ── Cross-test comparison ─────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("ALL TESTS COMPARISON (v1, v2, v3)")
    print(f"{'=' * 120}")

    comparison = []
    comparison.append({"test": "Synthetic benchmark", "arrays": 720,
                       "sizes": "10K–2M", "gap": 18.8,
                       "domain": "Synthetic (uniform, normal, lognormal, exp)"})

    if V1_PATH.exists():
        v1 = pd.read_parquet(V1_PATH)
        v1_tc = [c for c in v1.columns if c.startswith("time_")]
        v1_tm = v1[v1_tc].to_numpy()
        v1_vbs = v1_tm.min(axis=1).sum()
        v1_sbs = v1_tm[:, v1_tm.sum(axis=0).argmin()].sum()
        v1_gap = (v1_sbs - v1_vbs) / v1_vbs * 100
        comparison.append({"test": "Real-world v1 (F1 fastest lap)", "arrays": len(v1),
                           "sizes": f"{int(v1['n'].min())}–{int(v1['n'].max())}",
                           "gap": round(v1_gap, 1), "domain": "F1 telemetry (1 lap)"})

    if V2_PATH.exists():
        v2 = pd.read_parquet(V2_PATH)
        v2_tc = [c for c in v2.columns if c.startswith("time_")]
        v2_tm = v2[v2_tc].to_numpy()
        v2_vbs = v2_tm.min(axis=1).sum()
        v2_sbs = v2_tm[:, v2_tm.sum(axis=0).argmin()].sum()
        v2_gap = (v2_sbs - v2_vbs) / v2_vbs * 100
        comparison.append({"test": "Real-world v2 (F1 full race)", "arrays": len(v2),
                           "sizes": f"{int(v2['n'].min()):,}–{int(v2['n'].max()):,}",
                           "gap": round(v2_gap, 1), "domain": "F1 telemetry (all laps)"})

    comparison.append({"test": "Real-world v3 (Finance + Seismic)", "arrays": len(real_df),
                       "sizes": f"{int(real_df['n'].min()):,}–{int(real_df['n'].max()):,}",
                       "gap": round(gap, 1), "domain": "Stock, crypto, earthquake"})

    print(f"\n{'Test':<40} {'Arrays':>7} {'Size range':>16} {'VBS-SBS gap':>12} {'Domain':>40}")
    print("─" * 130)
    for c in comparison:
        print(f"{c['test']:<40} {c['arrays']:>7} {c['sizes']:>16} {c['gap']:>11.1f}% {c['domain']:>40}")

    # ── Save everything ───────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("SAVING RESULTS")
    print(f"{'=' * 120}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "real_world_v3_results.parquet"
    real_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved {parquet_path.relative_to(PROJECT_ROOT)}  ({len(real_df)} rows)")

    csv_path = OUTPUT_DIR / "real_world_v3_results.csv"
    real_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path.relative_to(PROJECT_ROOT)}  ({len(real_df)} rows)")

    config = {
        "version": "v3",
        "description": "Real-world test on financial markets + seismic data",
        "data_sources": {
            "stocks": {"tickers": STOCK_TICKERS, "period": "max", "interval": "1d"},
            "crypto": {"tickers": CRYPTO_TICKERS, "period": "max", "interval": "1d"},
            "earthquakes": {"source": "USGS API", "date_range": "2023-2025", "min_magnitude": 0.5},
        },
        "n_arrays": len(real_df),
        "n_features": len(_FEATURE_NAMES),
        "feature_names": _FEATURE_NAMES,
        "algorithms": list(ALGORITHMS.keys()),
        "n_max": N_MAX,
        "array_size_range": [int(real_df["n"].min()), int(real_df["n"].max())],
        "winner_counts": winner_counts.to_dict(),
        "vbs_sbs_gap_pct": round(gap, 2),
        "sbs_algorithm": sbs_name,
        "domain_breakdown": {
            domain: real_df[real_df["domain"] == domain]["best_algorithm"].value_counts().to_dict()
            for domain in ["stock", "crypto", "earthquake"]
        },
    }
    config_path = OUTPUT_DIR / "real_world_v3_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    print(f"  ✓ Saved {config_path.relative_to(PROJECT_ROOT)}")

    print(f"\n{'=' * 120}")
    print(f"DONE — v3 results saved to data/real_world_v3/")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
