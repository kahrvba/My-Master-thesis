# Adaptive Sorting Algorithm Selection — Master Thesis

## What This Project Does

Given a 1D numeric array at runtime, predict which sorting algorithm will be fastest — without running all of them. Extract cheap statistical features from the array, feed them to a selector model, get the best algorithm, sort.

## Problem Formulation

- Input: feature vector x ∈ R^16 (array characteristics computed in O(n))
- Output: a* = argmin_a T_a(x) — the algorithm with the lowest execution time for this array
- This is a single-step, stateless decision. No sequential dependency.
- Model approach: REGRESSION (predict time per algorithm, pick min) — not classification
  - Avoids hard-label class imbalance when some algorithms rarely win
  - Handles near-ties gracefully (if two algorithms are within 5%, either is fine)
  - Enables direct regret measurement: T_predicted - T_oracle

## Architecture Decided

Two-layer system, not two competing models:

```
              ┌─ size < threshold? ─→ just use Timsort (no selection overhead)
[Raw Array] ──┤
              └─ size ≥ threshold? ─→ [Feature Extraction] → [Selector] → [Sort]
```

The size threshold is a RESULT, not a parameter — determined by measuring when feature extraction cost exceeds selection savings. Reported in thesis.

### Layer 1: Offline Selector (XGBoost multi-output regression)
- Trained once on labeled synthetic data (features → predicted time per algorithm)
- Picks algorithm with lowest predicted time
- Frozen at deployment — never updates
- Purpose: strong static baseline + cold-start solution

### Layer 2: Online Selector (Contextual Bandit — LinUCB)
- Starts from XGBoost's knowledge
- Every time it sorts a real array, it observes real timing feedback
- Updates its model from that feedback — learns while deployed
- Purpose: adapts to unseen workload distributions without retraining
- This is the main thesis contribution

### Why Not DQN / Double DQN
- DQN is for sequential decision problems (multi-step MDPs)
- This problem is one-step: observe features → pick algorithm → done
- In a one-step MDP: next_state = state, done = True, always
- Bellman equation collapses to Q(s,a) = r — all RL machinery is dead weight
- Contextual bandits are the correct RL formulation for one-step context-dependent decisions
- The old notebooks (model-1.ipynb, mdoel-2.ipynb) used DQN — moved to old_models/ folder

## Sorting Algorithm Portfolio (3 algorithms, ALL C-level)

CRITICAL RULE: all implementations must be at C-speed (numpy ops, no Python loops in sort code). Mixing Python-loop sorts with C-level sorts makes timing comparisons meaningless — the model would learn language overhead, not algorithm characteristics.

| Algorithm  | Implementation                                        | Win rate (Step 2) |
|-----------|-------------------------------------------------------|-------------------|
| Introsort  | np.sort(kind='quicksort') — quicksort+heapsort hybrid | 33.3%             |
| Heapsort   | np.sort(kind='heapsort')                              | 44.4%             |
| Timsort    | np.sort(kind='stable') — adaptive, exploits runs      | 22.2%             |

Dropped after empirical evaluation:
- Counting sort — 0 wins in 720 benchmarks. Value range 0–30M forces 30M-entry allocation; always slower than comparison sorts at n≤2M
- Radix sort (LSD) — np.argsort per digit pass is O(n log n) per pass; 20x slower than introsort. Not a true radix sort without C-level implementation
- Bucket sort — never implemented; same problem as radix (per-bucket np.sort adds overhead)
- Bubble sort — O(n²), never competitive
- Insertion sort — can't implement at C-level without Cython
- Shell sort — same Cython problem
- Pure merge sort — numpy's 'mergesort' is actually Timsort since numpy 1.17

## Feature Set (v2 — 16 features)

Computed in O(n), no sorting required:

v1 core (5): length_norm, adj_sorted_ratio, duplicate_ratio, dispersion_ratio, runs_ratio

v2 additions (11): inversion_ratio, entropy_ratio, skewness_t, kurtosis_excess_t, longest_run_ratio, iqr_norm, mad_norm, top1_freq_ratio, top5_freq_ratio, outlier_ratio, mean_abs_diff_norm

All features bounded and validated — no NaN, no Inf, no duplicate sample_ids.

NOTE: inversion_ratio is the most expensive feature (merge-sort-based count, Python recursion). If feature importance analysis (Step 3) shows it's not useful, drop it — cuts feature extraction cost ~10x.

NOTE: outlier_ratio has very low variance in current data (max 0.073, std 0.012). May be dead weight. Keep for now, evaluate after Step 3.

## Synthetic Dataset

- 960 samples total (480 train / 240 val / 240 test) — STARTING POINT, will scale up
- 10 sizes: 1K to 50K
- 4 distributions: uniform, normal, lognormal, exponential
- 6 structures: random, nearly_sorted, reverse_sorted, few_unique, many_duplicates, runs
- 4 repeats per combination
- Stratified splits — no leakage, balanced across all factors
- Data lives in data/synthetic/raw/*.npy, splits in data/synthetic/splits/

## Bandit Evaluation: Designed Distribution Shift

The bandit's value only shows when XGBoost fails on unseen distributions. This must be designed into the experiment from day one, not retrofitted.

| Data Split    | Distributions         | XGBoost trained on it? | Bandit adapts to it? | Purpose                    |
|--------------|----------------------|----------------------|--------------------|-----------------------------|
| Train         | uniform, normal       | Yes                  | No                 | Train offline model          |
| Test A (in-dist) | uniform, normal    | No (held out)        | No                 | Evaluate XGBoost on known   |
| Test B (shift) | lognormal, exponential | NO                  | Yes (online)       | Evaluate bandit on unseen   |

XGBoost performs well on Test A, poorly on Test B.
Bandit starts poor on Test B, improves with feedback.
The regret curve on Test B is the bandit's thesis contribution.

NOTE: this requires restructuring data splits by distribution type, not just by sample_id. Plan into timing pipeline.

## Build Steps (ordered)

| Step | What                                                        | Status      |
|------|-------------------------------------------------------------|-------------|
| 1    | Synthetic data generation + feature extraction (v2)         | DONE        |
| 1b   | Pilot timing — verify algorithm landscape + VBS-SBS gap     | DONE        |
| 2    | Timing pipeline — 720 samples × 4→3 algorithms, record times | DONE        |
| 3    | XGBoost regressor — train, evaluate, feature importance, v1 vs v2 ablation | NOT STARTED |
| 4    | Baselines — random, always-best-single, decision tree, MLP  | NOT STARTED |
| 5    | LinUCB contextual bandit — online loop, regret curve on Test B | NOT STARTED |
| 6    | Comparison: XGBoost vs Bandit vs baselines                   | NOT STARTED |
| 7    | Real-world validation (F1 telemetry or other dataset)        | NOT STARTED |
| 8    | Package as Python library (import adaptive_sort; sort(arr))  | NOT STARTED |

## Thesis Contributions (5 defensible points)

1. Feature engineering — 16 cheap O(n) features that characterize sortability
2. Empirical benchmark — 3 C-level algorithms × 720 arrays, performance map showing when each wins
3. Offline selector — XGBoost regression, accuracy + regret analysis, feature importance
4. Online selector (NOVEL) — contextual bandit adapts to distribution shift without retraining
5. System — deployable Python library with two-tier threshold + adaptive selection

## Metrics

- Top-1 accuracy: did argmin of predicted times match argmin of actual times?
- Top-2 accuracy: was the oracle's pick in the model's top 2?
- Slowdown ratio: time_selected / time_oracle (1.0 = perfect)
- Regret: (time_selected - time_oracle) / time_oracle
- Regret curve (bandit): cumulative regret over time — shows convergence speed
- Feature extraction overhead: time to extract features vs time saved by selection

## Key Prior Work to Cite

- Rice (1976) — Algorithm Selection Problem (coined the concept)
- AutoFolio (Lindauer et al., 2015) — algorithm selection framework
- SATzilla (Xu et al., 2008-2012) — feature-based solver selection
- Learned Sort (Kristo et al., 2020) — ML-enhanced sorting
- Li & Mao (2009) — sorting selection via decision trees

## Stress-Tested Decisions (why we chose what we chose)

### Why regression not classification
- Classification: "which algorithm wins?" — class imbalance when some algorithms rarely win
- Regression: "how long does each take?" — pick min. Richer signal, no imbalance, near-ties handled.

### Why all C-level implementations
- Mixing Python-loop sorts with C sorts means the model learns language overhead, not algorithm performance
- numpy.sort(kind=...) gives us introsort, heapsort, Timsort at C level
- Radix and counting sort implemented via numpy vectorized ops (no Python loops)
- If we can't implement it at C-level without Cython, we don't include it

### Why two-tier threshold
- For small arrays, feature extraction costs more than just sorting
- Below threshold: skip selection, use Timsort
- Threshold value is measured and reported as a thesis result

### Why designed distribution shift
- If XGBoost gets 95% on i.i.d. test data, bandit has no room to improve
- Deliberately hold out distributions from training → bandit adapts to them online
- This is the experiment that proves the bandit's value

## Project Structure

```
scripts/
  generate_synthetic_dataset.py   — Step 1: data generation (DONE)
  extract_features.py             — Step 1: feature extraction (DONE)
  assess_dataset_quality.py       — Step 1: quality validation (DONE)
  pilot_timing.py                 — Step 1b: pilot timing v1 (DONE)
  pilot_timing_v2.py              — Step 1b: pilot timing v2 (DONE)
  vbs_sbs_gap.py                  — Step 1b: VBS vs SBS gap analysis (DONE)
  benchmark_algorithms.py         — Step 2: full timing pipeline (DONE)
  test_real_data.py               — v1: F1 fastest-lap test (35 arrays)
  test_real_data_v2.py            — v2: F1 full-race test (108 arrays)
  test_real_data_v3.py            — v3: Financial + seismic test (149 arrays)
  test_real_data_v4.py            — v4: Cross-domain combined benchmark (1,039 arrays)

data/
  synthetic/raw/                  — .npy arrays (Step 1, small scale)
  synthetic/splits/               — train/val/test sample_id CSVs (Step 1)
  features/                       — parquet feature files v2 (Step 1)
  benchmark/                      — Step 2 output (THE canonical dataset)
    all_samples.parquet           — 720 rows × (16 features + 4 timings + metadata)
    train.parquet                 — 216 samples (uniform + normal, 60%)
    val.parquet                   — 72 samples (uniform + normal, 20%)
    test_A.parquet                — 72 samples (uniform + normal, 20%)
    test_B.parquet                — 360 samples (lognormal + exponential, bandit eval)
    benchmark_config.json         — full pipeline config
  real_world/                     — v1 F1 results
  real_world_v2/                  — v2 F1 results
  real_world_v3/                  — v3 finance+seismic results
  real_world_v4/                  — v4 combined cross-domain results
    real_world_v4_combined.parquet — 1,039 rows (ALL data unified)
    real_world_v4_new_data.parquet — 62 new arrays (weather, NASA, earthquake ext, large-scale)

docs/
  feature-definitions.md          — Complete 16-feature reference
  feature-validation-report.md    — 214/214 feature tests passed
  feature-extraction-defense.md   — Feature extraction integrity audit
  real-world-f1-report.md         — v1 report
  real-world-f1-report-v2.md      — v2 report
  real-world-v3-report.md         — v3 report
  real-world-v4-report.md         — v4 cross-domain combined report
  vbs-sbs-gap-analysis.md         — VBS-SBS gap explanation + cross-domain strategy

artifacts/
  feature_config.json             — feature extraction config
  dataset_quality_report.json     — data quality checks
```

## Step 2 Results (Benchmark)

Run: 720 samples × 4 algorithms (introsort, heapsort, timsort, counting_sort)
Sizes: 10K, 50K, 100K, 500K, 1M, 2M | 4 distributions | 6 structures | 5 repeats
Runtime: 312s (5.2 min)

### Win Counts
| Algorithm      | Wins | Win Rate |
|---------------|------|----------|
| heapsort       | 320  | 44.4%    |
| introsort      | 240  | 33.3%    |
| timsort        | 160  | 22.2%    |
| counting_sort  | 0    | 0.0%     |

### VBS-SBS Gap
- VBS-SBS gap = 18.8% — "always heapsort" wastes 18.8% of total sorting time
- Consistent with pilot analysis (20.8% on 100K–10M only)
- Timsort dominates sorted/reverse_sorted (10–17× faster than heapsort)
- Introsort wins at medium sizes and some structures
- Heapsort wins random/large but loses catastrophically on pre-sorted data

### Key Findings
- counting_sort: DROPPED — 0 wins. Value range 0–30M makes bincount allocation too expensive at n≤2M
- 3 viable algorithms remain: introsort, heapsort, timsort
- Split balance good: win ratios consistent across train/val/test_A/test_B
- No NaN/Inf in features or timings — dataset is clean

### Dataset
- data/benchmark/all_samples.parquet — 720 rows, each with 16 features + 4 timing columns + metadata
- Splits by distribution: train (216, uniform+normal) / val (72) / test_A (72) / test_B (360, lognormal+exponential)

## Real-World Validation (v1–v4)

### v1: F1 Fastest Lap (35 arrays, n~700)
- timsort wins 77.1%, VBS-SBS gap 5.1%
- Small arrays — timsort dominance expected

### v2: F1 Full Race (108 arrays, 34K–1.13M)
- heapsort 42.6%, introsort 37.0%, timsort 20.4%, VBS-SBS gap 3.2%
- Genuine 3-way competition at scale

### v3: Financial + Seismic (149 arrays, 2K–309K)
- heapsort 64.4%, VBS-SBS gap 1.6%
- Low gap explained: homogeneous domains → one algo dominates within each

### v4: Cross-Domain Combined (1,039 arrays, 2K–2M)
- **Combines ALL previous data + new real-world data**
- New data: 5 cities weather (Open-Meteo), 23K NASA asteroids (JPL), 100K earthquakes (USGS), 10 large-scale generated arrays at 2M
- Headline: 17.5% combined gap, 12.0% "real-only" gap
- **HONEST READING: The 17.5% is dominated by 720 synthetic arrays (69% of data). The 12% includes 10 generated arrays. Every truly-real domain has gap under 3.1%.**
- Per-domain gaps: weather 0.4%, earthquake 0.4–0.8%, NASA 1.1%, stock 1.8%, crypto 2.5%, F1 3.1%
- Per-array margins remain large: 97.3% have >10%, 70.3% have >100%
- **Thesis framing: Contribution is per-array prediction accuracy + structural sensitivity, not aggregate gap inflation**
- Feature extraction validated: 0 NaN, 0 Inf on 62 new arrays

## Rules

- Always use venv: source venv/bin/activate
- Seed 42 for all randomness
- Parquet as canonical output format
- Features normalized on train split constants only
- All sort implementations must be C-level (numpy ops, no Python loops)
- No pretending simple things are novel — position contributions honestly
- Pilot before full pipeline — verify assumptions on small sample first
