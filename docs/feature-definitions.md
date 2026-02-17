# Feature Definitions — 16 Structural Features for Algorithm Selection

**Source of truth**: `scripts/feature_extraction.py`  
**Used by**: All benchmark, validation, and real-world test scripts  
**Input**: Any 1-D numeric NumPy array  
**Output**: A dict of 16 float values  

---

## Overview

The adaptive sorting-algorithm selector characterises each input array with 16 structural features before predicting which algorithm (introsort, heapsort, or timsort) will be fastest. These features capture **scale**, **sortedness**, **uniqueness**, **spread**, **structure**, **randomness**, **shape**, **outliers**, and **local order** — the properties that theory and empirical evidence show influence sorting performance.

All 16 features are extracted by a single function:

```python
from feature_extraction import extract_features, FEATURE_NAMES

features = extract_features(array, n_max=2_000_000, sample_id="my_array")
```

---

## Feature Table

| # | Name | Range | Group | Description |
|---|------|-------|-------|-------------|
| 1 | `length_norm` | [0, 1] | Scale | Normalised array length |
| 2 | `adj_sorted_ratio` | [0, 1] | Sortedness | Fraction of adjacent ascending pairs |
| 3 | `duplicate_ratio` | [0, 1] | Uniqueness | Proportion of duplicate values |
| 4 | `dispersion_ratio` | [0, 1] | Spread | Standard deviation relative to range |
| 5 | `runs_ratio` | [0, 1] | Structure | Number of monotonic runs per element |
| 6 | `inversion_ratio` | [0, 1] | Disorder | Normalised inversion count |
| 7 | `entropy_ratio` | [0, 1] | Randomness | Value-distribution entropy |
| 8 | `skewness_t` | ℝ | Shape | Log-transformed skewness |
| 9 | `kurtosis_excess_t` | ℝ | Shape | Log-transformed excess kurtosis |
| 10 | `longest_run_ratio` | [0, 1] | Structure | Longest monotonic run per element |
| 11 | `iqr_norm` | [0, 1] | Spread | Interquartile range relative to range |
| 12 | `mad_norm` | [0, 1] | Spread | Median absolute deviation relative to range |
| 13 | `top1_freq_ratio` | [0, 1] | Uniqueness | Most-frequent value's share |
| 14 | `top5_freq_ratio` | [0, 1] | Uniqueness | Top-5 most-frequent values' share |
| 15 | `outlier_ratio` | [0, 1] | Outliers | Fraction of values beyond 3σ |
| 16 | `mean_abs_diff_norm` | [0, 1] | Local order | Average step size relative to range |

---

## Detailed Definitions

### 1. `length_norm` — Scale

$$\texttt{length\_norm} = \text{clip}\!\left(\frac{n}{n_{\max}},\; 0,\; 1\right)$$

- **What**: Array length normalised by the global maximum ($n_{\max} = 2{,}000{,}000$).
- **Why it matters**: Algorithm overhead (e.g. timsort's run-detection pass) is amortised differently at different scales. Small arrays favour timsort; large arrays favour heapsort/introsort.
- **Boundary values**: 0 → empty array, 1 → maximum-length array.

---

### 2. `adj_sorted_ratio` — Sortedness

$$\texttt{adj\_sorted\_ratio} = \frac{1}{n-1}\sum_{i=1}^{n-1} \mathbf{1}[x_i \geq x_{i-1}]$$

- **What**: Fraction of consecutive element pairs that are in non-descending order.
- **Why it matters**: Timsort exploits existing order (ascending or descending runs). A high value means the array is already nearly sorted → timsort advantage. A value near 0.5 means random order → heapsort/introsort competitive.
- **Boundary values**: 1.0 → fully sorted, 0.0 → fully reverse-sorted, ~0.5 → random.
- **Real-world examples**: F1 distance channels ≈ 0.99, stock close prices ≈ 0.53, log returns ≈ 0.49.

---

### 3. `duplicate_ratio` — Uniqueness

$$\texttt{duplicate\_ratio} = 1 - \frac{n_{\text{unique}}}{n}$$

- **What**: Proportion of the array that consists of repeated values.
- **Why it matters**: High duplication reduces the effective comparison space. Timsort's galloping merge and introsort's partition can exploit equal-element clusters differently.
- **Boundary values**: 0.0 → all values unique, 1.0 → all values identical.
- **Real-world examples**: F1 brake channel ≈ 1.0 (binary 0/100), stock prices ≈ 0.02, earthquake magnitudes ≈ 0.05.

---

### 4. `dispersion_ratio` — Spread

$$\texttt{dispersion\_ratio} = \text{clip}\!\left(\frac{\sigma}{x_{\max} - x_{\min}},\; 0,\; 1\right)$$

- **What**: Standard deviation normalised by the value range.
- **Why it matters**: Low dispersion (values clustered near the mean) produces different comparison patterns than high dispersion (values spread across the full range). Affects partition balance in introsort.
- **Boundary values**: 0.0 → all values identical, ~0.29 → uniform distribution, higher → bimodal or heavy-tailed.
- **Theoretical reference**: For a uniform distribution, $\sigma / R = 1/\sqrt{12} \approx 0.289$.

---

### 5. `runs_ratio` — Structure

$$\texttt{runs\_ratio} = \text{clip}\!\left(\frac{\text{monotonic\_run\_count}}{n},\; 0,\; 1\right)$$

- **What**: Number of monotonic (ascending or descending) runs divided by array length.
- **Why it matters**: Timsort's core algorithm identifies and merges natural runs. Few long runs → timsort excels. Many short runs → data is essentially random → comparison-based algorithms (introsort, heapsort) are more competitive.
- **Boundary values**: 1/n → one single run (fully sorted or reversed), ~0.5 → random permutation (each run ≈ 2 elements), ~0.67 → maximally fragmented.
- **Computation**: Single O(n) pass tracking direction changes.
- **Real-world examples**: F1 DRS channel ≈ 0.003 (long constant runs), stock log returns ≈ 0.60 (near-random).

---

### 6. `inversion_ratio` — Disorder

$$\texttt{inversion\_ratio} = \text{clip}\!\left(\frac{\text{inversions}}{\binom{n}{2}},\; 0,\; 1\right)$$

- **What**: Number of pairs $(i, j)$ where $i < j$ but $x_i > x_j$, normalised by the maximum possible $\binom{n}{2}$.
- **Why it matters**: The classical measure of "how far from sorted" an array is. Directly proportional to the number of swaps needed by insertion sort. Complements `adj_sorted_ratio` by capturing **global** disorder, not just local.
- **Scalability**: For arrays > 10,000 elements, we subsample 2,000 uniformly-spaced indices (deterministic seed based on `sample_id`) and compute inversions on the subsample. This keeps cost $O(m \log m)$ instead of $O(n \log n)$.
- **Boundary values**: 0.0 → sorted, 1.0 → reverse-sorted, ~0.5 → random.
- **Algorithm**: Merge-sort-based inversion counting, $O(n \log n)$.

---

### 7. `entropy_ratio` — Randomness

$$\texttt{entropy\_ratio} = \text{clip}\!\left(\frac{H}{H_{\max}},\; 0,\; 1\right), \quad H = -\sum_{k=1}^{32} p_k \log_2 p_k, \quad H_{\max} = \log_2 32 = 5$$

- **What**: Shannon entropy of a 32-bin equal-width histogram, normalised by the maximum possible entropy.
- **Why it matters**: High entropy means values are spread uniformly across the range → no structure to exploit → heapsort/introsort. Low entropy means values are concentrated in a few bins → more duplicate-like behaviour.
- **Why 32 bins**: Balances resolution vs stability. Too few bins lose distribution shape; too many bins create sparse histograms on small arrays.
- **Boundary values**: 0.0 → all values in one bin, 1.0 → perfectly uniform across all 32 bins.
- **Real-world examples**: F1 DRS ≈ 0.05 (near-constant), stock log returns ≈ 0.70, earthquake magnitudes ≈ 0.55.

---

### 8. `skewness_t` — Shape

$$\texttt{skewness\_t} = \text{sign}(S) \cdot \ln(1 + |S|), \quad S = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \mu}{\sigma}\right)^3$$

- **What**: Log-transformed sample skewness. The $\text{sign} \cdot \log(1+|x|)$ transform compresses extreme values while preserving direction.
- **Why it matters**: Skewed distributions produce unbalanced partition trees in quicksort/introsort. Heapsort is invariant to skew.
- **Why log-transformed**: Raw skewness can range from $-\infty$ to $+\infty$. The signed-log transform maps it to a bounded, roughly symmetric range suitable for tree-based ML models.
- **Boundary values**: 0.0 → symmetric, positive → right-skewed (long right tail), negative → left-skewed.
- **Real-world examples**: Earthquake magnitudes ≈ 0.80 (Gutenberg-Richter right skew), stock log returns ≈ −0.36 (crash left skew), earthquake time gaps ≈ 5.33 (extreme right skew from aftershock clustering).

---

### 9. `kurtosis_excess_t` — Shape

$$\texttt{kurtosis\_excess\_t} = \text{sign}(K) \cdot \ln(1 + |K|), \quad K = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \mu}{\sigma}\right)^4 - 3$$

- **What**: Log-transformed excess kurtosis ($-3$ makes a normal distribution have $K = 0$).
- **Why it matters**: High kurtosis (leptokurtic) → more outliers → more branch mispredictions in comparison-based sorts. Affects introsort's pivot selection quality.
- **Boundary values**: 0.0 → normal-like tails, positive → heavy tails (leptokurtic), negative → light tails (platykurtic).
- **Real-world examples**: Earthquake time gaps ≈ 10.77 (extreme fat tails), stock returns ≈ 2.5 (fat tails from crashes/rallies), F1 speed ≈ 0.3 (near-normal).

---

### 10. `longest_run_ratio` — Structure

$$\texttt{longest\_run\_ratio} = \text{clip}\!\left(\frac{\ell_{\max}}{n},\; 0,\; 1\right)$$

- **What**: Length of the longest monotonic (ascending or descending) run, normalised by array length.
- **Why it matters**: Timsort's `minrun` parameter (typically 32–64) determines the minimum run length. If the longest natural run exceeds `minrun`, timsort gains efficiency by merging rather than insertion-sorting. This feature captures whether there is a **dominant** run in the data.
- **Boundary values**: 1.0 → entire array is one run (sorted/reversed), 2/n → all runs are length 2 (random), 1/n → alternating single elements.
- **Computed together with** `runs_ratio` in a single O(n) pass.
- **Real-world examples**: F1 distance ≈ 0.11 (long ascending segments), stock prices ≈ 0.001 (no dominant run).

---

### 11. `iqr_norm` — Spread

$$\texttt{iqr\_norm} = \text{clip}\!\left(\frac{Q_{75} - Q_{25}}{x_{\max} - x_{\min}},\; 0,\; 1\right)$$

- **What**: Interquartile range (middle 50% of data) normalised by the total range.
- **Why it matters**: Robust measure of spread that is insensitive to outliers (unlike `dispersion_ratio`). High IQR → values spread broadly → good partition balance for introsort. Low IQR → values clustered → many equal-element comparisons.
- **Boundary values**: 0.0 → all values identical, 0.5 → uniform distribution, near 1.0 → bimodal at extremes.
- **Complements**: `dispersion_ratio` (which uses standard deviation, sensitive to outliers).

---

### 12. `mad_norm` — Spread

$$\texttt{mad\_norm} = \text{clip}\!\left(\frac{\text{median}(|x_i - \tilde{x}|)}{x_{\max} - x_{\min}},\; 0,\; 1\right)$$

- **What**: Median Absolute Deviation from the median, normalised by the value range.
- **Why it matters**: The most robust spread measure (50% breakdown point). Captures whether the "typical" value is close to or far from the centre, independent of extreme outliers.
- **Boundary values**: 0.0 → all values identical, ~0.25 → uniform, higher → spread away from centre.
- **Complements**: `dispersion_ratio` (std-based, outlier-sensitive) and `iqr_norm` (quartile-based).

---

### 13. `top1_freq_ratio` — Uniqueness

$$\texttt{top1\_freq\_ratio} = \text{clip}\!\left(\frac{\max_v \text{count}(v)}{n},\; 0,\; 1\right)$$

- **What**: Frequency of the single most common value, as a fraction of total elements.
- **Why it matters**: A dominant value (e.g. brake=0 in F1 telemetry) creates long equal-element runs. Timsort handles these via galloping; heapsort sifts through them; introsort's partition may degenerate. This is one of the strongest discriminators between algorithms.
- **Boundary values**: 1/n → all unique, 1.0 → all values identical.
- **Real-world examples**: F1 brake ≈ 0.975 (nearly all zeros), stock prices ≈ 0.001 (all unique).

---

### 14. `top5_freq_ratio` — Uniqueness

$$\texttt{top5\_freq\_ratio} = \text{clip}\!\left(\frac{\sum_{k=1}^{5} \text{count}(v_k)}{n},\; 0,\; 1\right)$$

- **What**: Combined frequency of the 5 most common values, as a fraction of total elements.
- **Why it matters**: Captures whether the array is dominated by a small set of categorical-like values (e.g. F1 gear positions: 1–8) even when no single value dominates. High values indicate near-categorical data that may benefit from different algorithmic strategies.
- **Boundary values**: 5/n → all unique, 1.0 → at most 5 distinct values.
- **Real-world examples**: F1 nGear ≈ 1.0 (only 8 gear values), stock prices ≈ 0.005.

---

### 15. `outlier_ratio` — Outliers

$$\texttt{outlier\_ratio} = \text{clip}\!\left(\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[\left|\frac{x_i - \mu}{\sigma}\right| > 3\right],\; 0,\; 1\right)$$

- **What**: Fraction of values more than 3 standard deviations from the mean.
- **Why it matters**: Outliers affect quicksort pivot selection (bad pivots → unbalanced partitions → $O(n^2)$ risk, which introsort mitigates by switching to heapsort). Also affects timsort's run structure since outliers break natural runs.
- **Boundary values**: 0.0 → no outliers (or constant array), expected ~0.003 for normal distribution.
- **Real-world examples**: Earthquake data ≈ 0.047, F1 telemetry ≈ 0.006.

---

### 16. `mean_abs_diff_norm` — Local Order

$$\texttt{mean\_abs\_diff\_norm} = \text{clip}\!\left(\frac{\frac{1}{n-1}\sum_{i=1}^{n-1}|x_{i+1} - x_i|}{x_{\max} - x_{\min}},\; 0,\; 1\right)$$

- **What**: Average absolute difference between adjacent elements, normalised by the value range.
- **Why it matters**: Captures **local smoothness**. Low values → adjacent elements are similar (smooth signal, natural runs) → timsort advantage. High values → adjacent elements vary wildly (noisy, random) → no run structure to exploit.
- **Boundary values**: 0.0 → constant array, low → smooth/sorted, ~0.33 → random uniform, higher → adversarial oscillation.
- **Real-world examples**: F1 speed ≈ 0.003 (smooth sensor signal), stock log returns ≈ 0.025 (moderate noise).

---

## Feature Groups

| Group | Features | What It Captures |
|-------|----------|-----------------|
| **Scale** | `length_norm` | Array size relative to maximum |
| **Sortedness** | `adj_sorted_ratio` | Local ascending order |
| **Disorder** | `inversion_ratio` | Global out-of-order-ness |
| **Structure** | `runs_ratio`, `longest_run_ratio` | Monotonic run count and size |
| **Uniqueness** | `duplicate_ratio`, `top1_freq_ratio`, `top5_freq_ratio` | Value repetition patterns |
| **Spread** | `dispersion_ratio`, `iqr_norm`, `mad_norm` | How values are spread (3 robustness levels) |
| **Randomness** | `entropy_ratio` | Value-distribution uniformity |
| **Shape** | `skewness_t`, `kurtosis_excess_t` | Distribution asymmetry and tail weight |
| **Outliers** | `outlier_ratio` | Extreme value prevalence |
| **Local order** | `mean_abs_diff_norm` | Adjacent-element smoothness |

---

## Computational Complexity

| Feature | Cost | Notes |
|---------|------|-------|
| `length_norm` | O(1) | Array size is known |
| `adj_sorted_ratio` | O(n) | Single diff pass |
| `duplicate_ratio` | O(n log n) | `np.unique` sort |
| `dispersion_ratio` | O(n) | Mean + std |
| `runs_ratio` | O(n) | Single pass |
| `longest_run_ratio` | O(n) | Same pass as `runs_ratio` |
| `inversion_ratio` | O(n log n) or O(m log m) | Merge-sort; subsampled to m=2000 for n>10K |
| `entropy_ratio` | O(n) | Histogram binning |
| `skewness_t` | O(n) | Z-score moments |
| `kurtosis_excess_t` | O(n) | Same pass as skewness |
| `iqr_norm` | O(n) | Percentile computation |
| `mad_norm` | O(n) | Median + abs deviation |
| `top1_freq_ratio` | O(n log n) | Reuses `np.unique` from `duplicate_ratio` |
| `top5_freq_ratio` | O(n log n) | Same |
| `outlier_ratio` | O(n) | Z-score thresholding |
| `mean_abs_diff_norm` | O(n) | Reuses diffs from `adj_sorted_ratio` |

**Total**: O(n log n) dominated by `np.unique` and `inversion_ratio`. For n > 10K, inversion subsampling drops that cost to O(n log n) from `np.unique` only.

---

## Why These 16 Features?

### Design Principles

1. **Theoretically motivated**: Each feature maps to a known algorithmic property. Timsort exploits runs (features 2, 5, 10, 16), heapsort is invariant to input structure (no feature helps it specifically — it wins by default when nothing else gives an advantage), introsort's quicksort phase depends on pivot quality (features 4, 8, 9, 15).

2. **Normalised**: 14 of 16 features are bounded [0, 1], making them directly usable by tree-based and linear models without scaling. The 2 unbounded features (skewness, kurtosis) are log-compressed.

3. **Non-redundant but complementary**: The 3 spread features (`dispersion_ratio`, `iqr_norm`, `mad_norm`) have different robustness levels (outlier-sensitive → quartile → median). The 3 uniqueness features (`duplicate_ratio`, `top1_freq_ratio`, `top5_freq_ratio`) capture different concentration patterns.

4. **Cheap**: Total extraction cost is $O(n \log n)$ — negligible compared to the sort itself at the same complexity.

5. **Domain-agnostic**: Validated on synthetic data (720 arrays), F1 telemetry (143 arrays), stock market (124 arrays), cryptocurrency (18 arrays), and earthquake catalogs (7 arrays) with physically correct values in every domain.

### What Each Feature Predicts

| If this feature is... | High | Low |
|-----------------------|------|-----|
| `adj_sorted_ratio` | Timsort advantage (natural runs) | No run structure to exploit |
| `duplicate_ratio` | Equal-element handling matters | All comparisons are strict |
| `runs_ratio` | Random (many short runs) | Ordered (few long runs) → timsort |
| `inversion_ratio` | Far from sorted → heapsort/introsort | Nearly sorted → timsort |
| `entropy_ratio` | Uniform spread → heapsort/introsort | Concentrated → special structure |
| `longest_run_ratio` | Dominant run exists → timsort | No exploitable run |
| `mean_abs_diff_norm` | Noisy/random → heapsort/introsort | Smooth → timsort |

---

## Validation Summary

| Test | Arrays | Size Range | Features Within Benchmark Range | Sanity Checks |
|------|--------|------------|---:|---:|
| Unit tests (`validate_features.py`) | 214 tests | 1–50K | N/A | 214/214 ✓ |
| Synthetic benchmark | 720 | 10K–2M | 16/16 (defines range) | All ✓ |
| F1 telemetry v1 | 35 | ~700 | 12/16 | All ✓ |
| F1 telemetry v2 | 108 | 34K–1.13M | 12/16 | All ✓ |
| Financial + Seismic v3 | 149 | 2K–309K | 14/16 | All ✓ |

Features outside benchmark range in real-world tests are physically explainable (fat-tailed financial returns, earthquake power laws) and handled by the LinUCB bandit's online exploration.

---

## Implementation Reference

**File**: `scripts/feature_extraction.py`  
**Function**: `extract_features(values, n_max, sample_id) → dict`  
**Constants**: `FEATURE_NAMES` (list of 16 strings), `SEED=42`, `EPS=1e-12`  
**Helpers**: `_monotonic_run_stats()`, `_inversion_count_merge()`, `_signed_log1p()`  

All other scripts import from this single file:
```
benchmark_algorithms.py  →  from feature_extraction import ...
test_real_data.py        →  from feature_extraction import ...
test_real_data_v2.py     →  from feature_extraction import ...
test_real_data_v3.py     →  from feature_extraction import ...
validate_features.py     →  from feature_extraction import ...
```
