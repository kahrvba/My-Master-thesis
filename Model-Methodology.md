First Step: Feature Extraction

- Goal: start with easy/fast features first (`v1`), then add harder features (`v2`) only if results improve.
- Input sample format: `sample_id`, `source`, `values` (1D numeric array).
- Deterministic rule: use seed `42` for any sampling step.

- Feature set `v1` (default, cheap):
- `length_norm = n / n_max_train`
- `adj_sorted_ratio = count(values[i] <= values[i+1]) / (n - 1)`
- `duplicate_ratio = 1 - (unique_count / n)`
- `dispersion_ratio = std(values) / (max(values) - min(values) + 1e-12)`
- `runs_ratio = monotonic_runs / n`

- Feature set `v2` (optional, only if needed):
- `inversion_ratio` (sampled for large arrays)
- `entropy_ratio` (histogram-based)
- `skewness_t = sign(skewness) * log1p(abs(skewness))`
- `kurtosis_excess_t = sign(kurtosis_excess) * log1p(abs(kurtosis_excess))`
- `longest_run_ratio` (longest monotonic run / n)
- `iqr_norm` (`(p75 - p25) / (max - min + 1e-12)`)
- `mad_norm` (`median(abs(x - median(x))) / (max - min + 1e-12)`)
- `top1_freq_ratio` (most frequent value count / n)
- `top5_freq_ratio` (sum of top-5 value counts / n)
- `outlier_ratio` (fraction with `|z| > 3`)
- `mean_abs_diff_norm` (`mean(abs(diff(x))) / (max - min + 1e-12)`)

- `v2` inversion rule:
- Large-array threshold: `n > 10000`
- Sample size: `m = min(2000, n)`
- If `n <= 10000`: exact inversion ratio
- If `n > 10000`: sampled inversion estimate with seed `42`

- Normalization constants fit on train split only: `n_max_train`.
- Reuse same constants for validation and test.
- Clip normalized features to `[0.0, 1.0]` when needed.

- Output file format: `parquet`.
- Output files:
- `data/features/train_features.parquet`
- `data/features/val_features.parquet`
- `data/features/test_features.parquet`
- Save config in `artifacts/feature_config.json` with `feature_set: v1` or `feature_set: v2`.

- Required output columns for `v1`:
- `sample_id`
- `length_norm`
- `adj_sorted_ratio`
- `duplicate_ratio`
- `dispersion_ratio`
- `runs_ratio`

- If `v2` is enabled, append:
- `inversion_ratio`
- `entropy_ratio`
- `skewness_t`
- `kurtosis_excess_t`
- `longest_run_ratio`
- `iqr_norm`
- `mad_norm`
- `top1_freq_ratio`
- `top5_freq_ratio`
- `outlier_ratio`
- `mean_abs_diff_norm`

- Promotion rule from `v1` to `v2`:
- Move to `v2` only if ablation shows clear gain with acceptable overhead.

- Sanity checks:
- No NaN or inf in any feature column.
- No duplicate `sample_id` rows.
- Feature values inside expected range after normalization.
- Re-running extraction with same seed produces identical outputs.
