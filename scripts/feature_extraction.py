#!/usr/bin/env python3
"""
Feature Extraction — Single Source of Truth
=============================================
This module defines the 16 structural features used by the adaptive
sorting-algorithm selector.  Every script in the project imports from
HERE — there is exactly ONE implementation.

Features (16 total)
────────────────────
 #  Name                  Range     Group           Description
 1  length_norm           [0, 1]    Scale           n / n_max (clipped)
 2  adj_sorted_ratio      [0, 1]    Sortedness      Fraction of adjacent pairs in ascending order
 3  duplicate_ratio       [0, 1]    Uniqueness      1 − (n_unique / n)
 4  dispersion_ratio      [0, 1]    Spread          std / value_range (clipped)
 5  runs_ratio            [0, 1]    Structure       Monotonic-run count / n
 6  inversion_ratio       [0, 1]    Disorder        Normalised inversion count (subsampled >10K)
 7  entropy_ratio         [0, 1]    Randomness      32-bin histogram entropy / log₂(32)
 8  skewness_t            ℝ         Shape           sign(S)·log₁(1+|S|), S = E[(x−μ)³/σ³]
 9  kurtosis_excess_t     ℝ         Shape           sign(K)·log₁(1+|K|), K = E[(x−μ)⁴/σ⁴]−3
10  longest_run_ratio     [0, 1]    Structure       Longest monotonic run / n
11  iqr_norm              [0, 1]    Spread          (Q75 − Q25) / value_range
12  mad_norm              [0, 1]    Spread          median(|x − median|) / value_range
13  top1_freq_ratio       [0, 1]    Uniqueness      Most-frequent count / n
14  top5_freq_ratio       [0, 1]    Uniqueness      Top-5 most-frequent counts / n
15  outlier_ratio         [0, 1]    Outliers        Fraction with |z| > 3
16  mean_abs_diff_norm    [0, 1]    Local order     mean(|diff|) / value_range

Usage
─────
    from feature_extraction import extract_features, FEATURE_NAMES
"""

from __future__ import annotations

import hashlib

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────

SEED = 42          # Global seed (used for deterministic inversion subsampling)
EPS  = 1e-12       # Numerical-stability epsilon

FEATURE_NAMES: list[str] = [
    "length_norm",
    "adj_sorted_ratio",
    "duplicate_ratio",
    "dispersion_ratio",
    "runs_ratio",
    "inversion_ratio",
    "entropy_ratio",
    "skewness_t",
    "kurtosis_excess_t",
    "longest_run_ratio",
    "iqr_norm",
    "mad_norm",
    "top1_freq_ratio",
    "top5_freq_ratio",
    "outlier_ratio",
    "mean_abs_diff_norm",
]

# Backward-compatible alias used by older scripts
_FEATURE_NAMES = FEATURE_NAMES


# ─── Helper Functions ────────────────────────────────────────────────────

def _monotonic_run_stats(values: np.ndarray) -> tuple[int, int]:
    """Count monotonic runs and longest run length.  O(n)."""
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
    """Merge-sort-based inversion count.  O(n log n)."""
    n = len(arr)
    if n <= 1:
        return 0
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]
    inv_l = _inversion_count_merge(left)
    inv_r = _inversion_count_merge(right)

    merged: list[float] = []
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
    """Signed log-transform: sign(x) · log(1 + |x|)."""
    return float(np.sign(x) * np.log1p(abs(x)))


# ─── Main Feature Extraction ─────────────────────────────────────────────

def extract_features(values: np.ndarray, n_max: float, sample_id: str) -> dict:
    """
    Extract all 16 structural features from a 1-D numeric array.

    Parameters
    ----------
    values : np.ndarray
        The raw 1-D array to characterise.
    n_max : float
        Global maximum array length (for length_norm normalisation).
    sample_id : str
        Identifier used for deterministic subsampling seed.

    Returns
    -------
    dict
        {feature_name: float} for all 16 features.
    """
    n = int(values.size)
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    vals = values.astype(np.float64, copy=False)
    vmin = float(vals.min())
    vmax = float(vals.max())
    value_range = vmax - vmin
    std = float(vals.std())
    mean = float(vals.mean())
    med = float(np.median(vals))

    # ── 1. length_norm ──
    length_norm = float(np.clip(n / (n_max + EPS), 0.0, 1.0))

    # ── 2. adj_sorted_ratio ──
    diffs = np.diff(vals)
    adj_sorted_ratio = float(np.mean(diffs >= 0)) if n > 1 else 1.0

    # ── 3. duplicate_ratio ──
    uniq_vals, uniq_counts = np.unique(vals, return_counts=True)
    n_unique = int(uniq_vals.size)
    duplicate_ratio = float(1.0 - n_unique / n)

    # ── 4. dispersion_ratio ──
    dispersion_ratio = float(np.clip(std / (value_range + EPS), 0.0, 1.0))

    # ── 5. runs_ratio  &  10. longest_run_ratio ──
    runs_count, longest_run = _monotonic_run_stats(values)
    runs_ratio = float(np.clip(runs_count / n, 0.0, 1.0))
    longest_run_ratio = float(np.clip(longest_run / n, 0.0, 1.0))

    # ── 6. inversion_ratio (expensive — subsample for large arrays) ──
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

    # ── 7. entropy_ratio (32-bin histogram) ──
    if value_range < EPS:
        entropy_ratio = 0.0
    else:
        hist, _ = np.histogram(vals, bins=32)
        p = hist[hist > 0].astype(np.float64)
        p /= p.sum()
        ent = -np.sum(p * np.log2(p))
        entropy_ratio = float(np.clip(ent / np.log2(32.0), 0.0, 1.0))

    # ── 8. skewness_t  &  9. kurtosis_excess_t ──
    if std < EPS:
        skewness_t = 0.0
        kurtosis_excess_t = 0.0
    else:
        z = (vals - mean) / (std + EPS)
        skewness_t = _signed_log1p(float(np.mean(z**3)))
        kurtosis_excess_t = _signed_log1p(float(np.mean(z**4) - 3.0))

    # ── 11. iqr_norm ──
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr_norm = float(np.clip((q75 - q25) / (value_range + EPS), 0.0, 1.0))

    # ── 12. mad_norm ──
    mad_norm = float(np.clip(
        np.median(np.abs(vals - med)) / (value_range + EPS), 0.0, 1.0
    ))

    # ── 13. top1_freq_ratio  &  14. top5_freq_ratio ──
    sorted_counts = np.sort(uniq_counts)[::-1]
    top1_freq_ratio = float(np.clip(sorted_counts[0] / n, 0.0, 1.0))
    top5_freq_ratio = float(np.clip(sorted_counts[:5].sum() / n, 0.0, 1.0))

    # ── 15. outlier_ratio ──
    if std < EPS:
        outlier_ratio = 0.0
    else:
        z_abs = np.abs((vals - mean) / (std + EPS))
        outlier_ratio = float(np.clip(np.mean(z_abs > 3.0), 0.0, 1.0))

    # ── 16. mean_abs_diff_norm ──
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
