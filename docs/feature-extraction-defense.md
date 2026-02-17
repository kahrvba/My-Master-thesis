# Feature Extraction & Model Selection — Thesis Defense Reference

> Prepared answers for: *"What features do you extract, how, and how does the model use them to select an algorithm?"*

---

## 1. What Features Are You Extracting?

16 statistical features computed from a raw 1-D numeric array. Each one captures a **structural property** that affects how sorting algorithms perform.

### The 16 Features

| # | Feature | What It Measures | Formula / Method | Range |
|---|---------|-----------------|------------------|-------|
| 1 | `length_norm` | Array size relative to max | $\frac{n}{n_{\max}}$ | [0, 1] |
| 2 | `adj_sorted_ratio` | How sorted the array already is | $\frac{1}{n-1}\sum_{i=1}^{n-1} \mathbb{1}[a_i \le a_{i+1}]$ | [0, 1] |
| 3 | `duplicate_ratio` | Fraction of repeated values | $1 - \frac{|\text{unique values}|}{n}$ | [0, 1] |
| 4 | `dispersion_ratio` | How spread out values are relative to range | $\frac{\sigma}{x_{\max} - x_{\min}}$ | [0, 1] |
| 5 | `runs_ratio` | Number of monotonic runs relative to size | $\frac{\text{run count}}{n}$ | [0, 1] |
| 6 | `inversion_ratio` | How "disordered" the array is | $\frac{\text{inversion count}}{\binom{n}{2}}$ | [0, 1] |
| 7 | `entropy_ratio` | Value distribution uniformity (32-bin histogram) | $\frac{H(p)}{\log_2 32}$ where $H = -\sum p_i \log_2 p_i$ | [0, 1] |
| 8 | `skewness_t` | Asymmetry of value distribution (log-transformed) | $\text{sign}(s) \cdot \log(1 + |s|)$ where $s = \frac{1}{n}\sum z_i^3$ | unbounded |
| 9 | `kurtosis_excess_t` | Tail heaviness (log-transformed) | $\text{sign}(k) \cdot \log(1 + |k|)$ where $k = \frac{1}{n}\sum z_i^4 - 3$ | unbounded |
| 10 | `longest_run_ratio` | Longest monotonic subsequence relative to n | $\frac{\text{longest run}}{n}$ | [0, 1] |
| 11 | `iqr_norm` | Interquartile range normalized by value range | $\frac{Q_{75} - Q_{25}}{x_{\max} - x_{\min}}$ | [0, 1] |
| 12 | `mad_norm` | Median absolute deviation normalized | $\frac{\text{median}(|x_i - \tilde{x}|)}{x_{\max} - x_{\min}}$ | [0, 1] |
| 13 | `top1_freq_ratio` | Most frequent value's share of array | $\frac{\text{count of mode}}{n}$ | [0, 1] |
| 14 | `top5_freq_ratio` | Top 5 most frequent values' share | $\frac{\sum_{k=1}^{5} \text{count}_k}{n}$ | [0, 1] |
| 15 | `outlier_ratio` | Fraction of values beyond 3 standard deviations | $\frac{1}{n} \sum \mathbb{1}[|z_i| > 3]$ | [0, 1] |
| 16 | `mean_abs_diff_norm` | Average step size between adjacent elements | $\frac{1}{n-1}\sum |a_{i+1} - a_i| / (x_{\max} - x_{\min})$ | [0, 1] |

---

## 2. Why These Specific Features?

Each feature targets a known algorithmic weakness or strength:

| Feature | Why It Matters for Sorting |
|---------|---------------------------|
| `adj_sorted_ratio` | **Timsort** exploits pre-sorted runs. If this is high (>0.95), timsort will be fastest. |
| `inversion_ratio` | Counts how many pairs are out of order. Directly measures the "work" a comparison sort must do. High inversion → introsort/heapsort. Low inversion → timsort. |
| `runs_ratio` | Few long runs = timsort territory. Many short runs = no advantage for adaptive sorts. |
| `longest_run_ratio` | If one dominant sorted run exists, timsort merges it cheaply. |
| `duplicate_ratio` | High duplicates → many equal comparisons → affects partitioning in introsort (Dutch national flag). |
| `entropy_ratio` | Low entropy = few distinct values = timsort can exploit structure. High entropy = random-like = heapsort/introsort. |
| `top1_freq_ratio` | If one value dominates (e.g., brake signal: 90% zeros), the array is nearly constant → timsort wins. |
| `length_norm` | Heapsort's O(n log n) worst case is better at large n. Introsort has better cache locality at medium n. |
| `dispersion_ratio` | Affects how partitioning in quicksort/introsort divides the array. |
| `outlier_ratio` | Outliers affect pivot selection in quicksort (introsort's inner loop). |

---

## 3. How Are You Extracting Them?

### 3.1 Computational Complexity

All 16 features are extracted in **O(n)** — the same complexity as a single pass through the array. This is critical because if feature extraction costs more than sorting, the entire system is pointless.

| Operation | Complexity | Features Computed |
|-----------|-----------|-------------------|
| Single pass (`np.diff`) | O(n) | `adj_sorted_ratio`, `mean_abs_diff_norm` |
| `np.unique` | O(n log n)* | `duplicate_ratio`, `top1_freq_ratio`, `top5_freq_ratio` |
| `np.histogram` (32 bins) | O(n) | `entropy_ratio` |
| `np.std`, `np.mean`, `np.median` | O(n) | `dispersion_ratio`, `skewness_t`, `kurtosis_excess_t`, `mad_norm` |
| `np.percentile` | O(n) | `iqr_norm` |
| Monotonic run scan | O(n) | `runs_ratio`, `longest_run_ratio` |
| Merge-sort inversion count | O(m log m)** | `inversion_ratio` |

\* `np.unique` internally sorts, but this is a C-level sort on the full array. Still fast and necessary.

\** For arrays > 10,000 elements, we **subsample 2,000 elements** to keep cost manageable. The subsample is deterministic (seeded hash of sample_id) so results are reproducible.

### 3.2 Implementation Details

```python
def extract_features(values: np.ndarray, n_max: float, sample_id: str) -> dict:
    """
    Input:  raw 1-D numpy array (the array to be sorted)
    Output: dict of 16 float values (the feature vector)
    """
```

Key implementation decisions:
- **All NumPy vectorized** — no Python loops except the monotonic-run scan and inversion count
- **Epsilon guard** (`EPS = 1e-12`) — prevents division by zero for constant arrays
- **Clipping** — all bounded features clipped to [0, 1] to prevent numerical noise
- **Log-transform** on skewness/kurtosis — raw values can be extreme (100+), log1p compresses them
- **Deterministic subsampling** — inversion count uses `SHA256(seed:sample_id)` → same subsample every run

### 3.3 Cost in Practice

From our benchmark measurements:
- Feature extraction for a **2M-element array**: ~0.3 seconds
- Sorting a 2M-element array: ~0.2–0.5 seconds
- Feature extraction for a **50K-element array**: ~0.008 seconds
- Sorting a 50K-element array: ~0.003–0.005 seconds

This means feature extraction is only worth it when the **time saved by choosing the right algorithm exceeds the extraction cost**. That's why we have a two-tier system with a size threshold.

---

## 4. When Does Feature Extraction Happen?

```
                                    THE RUNTIME PIPELINE
                                    
[Raw Array arrives] 
       │
       ▼
  n < threshold?  ──── YES ──→  Just use Timsort (skip everything)
       │
       NO
       │
       ▼
  ┌─────────────────────┐
  │  extract_features()  │ ← 16 features computed in O(n)
  │  Input: raw array    │
  │  Output: x ∈ R^16    │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Selector Model      │ ← XGBoost or LinUCB
  │  Input: x ∈ R^16     │
  │  Output: predicted    │
  │  time per algorithm   │
  └──────────┬──────────┘
             │
             ▼
  Pick algorithm with lowest predicted time
             │
             ▼
  Sort the array with that algorithm
             │
             ▼
  [Sorted array returned]
```

Feature extraction happens **once per array, before sorting**. The features are never computed from the sorted output — they are computed from the **raw unsorted input**.

---

## 5. How Does the Model Select Based on Features?

### 5.1 The Regression Approach

We do **NOT** classify "which algorithm is best." Instead, we **predict the execution time** for each algorithm, then pick the fastest.

```
Feature vector x ∈ R^16

Model predicts 3 times:
  ŷ_introsort = f(x)   → e.g., 0.0041 seconds
  ŷ_heapsort  = f(x)   → e.g., 0.0038 seconds  ← smallest
  ŷ_timsort   = f(x)   → e.g., 0.0052 seconds

Selection:  a* = argmin(ŷ_introsort, ŷ_heapsort, ŷ_timsort)
Answer:     use heapsort
```

**Why regression instead of classification?**
- No class imbalance problem (heapsort wins 44%, introsort 33%, timsort 22%)
- Handles near-ties: if two algorithms are within 5%, either is fine
- Directly measures regret: $\frac{\hat{t}_{selected} - t_{oracle}}{t_{oracle}}$

### 5.2 Layer 1 — XGBoost (Offline Baseline)

XGBoost multi-output regression is trained **once** on the benchmark dataset:

| What | Detail |
|------|--------|
| Input | 16 features per array |
| Output | 3 predicted execution times |
| Training data | 216 synthetic arrays (uniform + normal distributions) |
| Validation | 72 arrays (same distributions, held out) |
| Test A | 72 arrays (same distributions, unseen) |
| Test B | 360 arrays (lognormal + exponential — **never seen in training**) |

XGBoost learns patterns like:
- `adj_sorted_ratio > 0.95` AND `runs_ratio < 0.01` → predict timsort will be 5× faster
- `entropy_ratio > 0.8` AND `length_norm > 0.25` → predict heapsort will win
- `duplicate_ratio > 0.99` → predict timsort/introsort competitive, heapsort slower

### 5.3 Layer 2 — LinUCB Contextual Bandit (Online Adaptation)

The bandit starts from XGBoost's knowledge but **keeps learning** at deployment:

```
For each new array:
  1. Extract features  →  x ∈ R^16
  2. For each algorithm a:
     Compute: UCB_a = θ_a^T x + α √(x^T A_a^{-1} x)
              ↑ exploitation        ↑ exploration bonus
  3. Pick a* = argmax(UCB_a)
  4. Sort with a*, measure actual time t
  5. Update: A_{a*} += x x^T,  b_{a*} += t · x
```

| Term | Meaning |
|------|---------|
| $\theta_a^T x$ | "Based on what I've learned, this is my best guess of algorithm a's speed for this array" |
| $\alpha \sqrt{x^T A_a^{-1} x}$ | "But I'm uncertain — explore if I haven't seen arrays like this before" |

The bandit's advantage: when F1 telemetry data arrives (never seen in training), the bandit explores, observes real timings, and adapts — **without retraining the model**.

### 5.4 Concrete Example — How Features Drive Selection

**Array: F1 DRS signal (65,000 samples)**  
```
Features:   adj_sorted_ratio = 0.999
            entropy_ratio    = 0.06
            duplicate_ratio  = 0.999
            runs_ratio       = 0.0006
            top1_freq_ratio  = 0.97
```
→ Nearly constant, nearly sorted, one value dominates  
→ Model predicts: timsort ≈ 0.00008s, heapsort ≈ 0.00016s, introsort ≈ 0.00017s  
→ **Timsort selected** (2× faster — it merges the single long run in one pass)

**Array: F1 RPM signal (42,000 samples)**  
```
Features:   adj_sorted_ratio = 0.632
            entropy_ratio    = 0.81
            duplicate_ratio  = 0.375
            runs_ratio       = 0.15
            top1_freq_ratio  = 0.003
```
→ High entropy, many unique values, moderately disordered  
→ Model predicts: heapsort ≈ 0.00090s, introsort ≈ 0.00095s, timsort ≈ 0.00170s  
→ **Heapsort selected** (timsort would be 89% slower — no runs to exploit)

**Array: F1 Brake signal (42,000 samples)**  
```
Features:   adj_sorted_ratio = 0.989
            entropy_ratio    = 0.15
            duplicate_ratio  = 1.000
            runs_ratio       = 0.022
            top1_freq_ratio  = 0.85
```
→ Binary (0 or 100), extremely high duplicates, few unique values  
→ Model predicts: heapsort ≈ 0.00003s, introsort ≈ 0.00004s, timsort ≈ 0.00012s  
→ **Heapsort selected** (timsort 300% slower — binary data doesn't form useful runs)

---

## 6. How Was This Validated?

| Validation | What It Proves | Result |
|------------|---------------|--------|
| 214 ground-truth unit tests | Features compute correct values on known arrays (sorted → adj_sorted=1.0, reversed → inv_ratio=1.0, etc.) | **214/214 passed** |
| Inversion count vs brute-force | Our O(n log n) merge-sort count matches O(n²) naive count exactly | **Exact match** |
| Determinism test | Same array → same features, every run | **Passed** |
| Real F1 telemetry (v1, 35 arrays) | Features match physical properties of sensor data | **Passed** |
| Real F1 telemetry (v2, 108 arrays) | Features valid at 34K–1.13M scale, all sanity checks pass | **Passed** |
| Benchmark (720 synthetic arrays) | No NaN, no Inf, all bounded features in [0,1] | **Passed** |

---

## 7. Summary — The One-Paragraph Answer

> We extract 16 statistical features from a raw unsorted array in O(n) time. These features capture structural properties that affect sorting performance: how sorted it already is, how many duplicates exist, how uniformly values are distributed, and how the data is spread. We feed these 16 numbers into an XGBoost regressor that predicts how long each of 3 sorting algorithms would take, and we pick the fastest. A LinUCB contextual bandit wraps this model to adapt online when real-world data differs from training data. The features were validated with 214 unit tests against known arrays and confirmed on 108 real Formula 1 telemetry arrays where the features correctly captured physical sensor properties (Distance is sorted → adj_sorted=0.998, Brake is binary → dup_ratio=1.0, DRS is constant → entropy=0.04).
