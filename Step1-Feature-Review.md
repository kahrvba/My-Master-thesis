Step 1 Feature Review and Fixes

- Finding: `adj_sorted_ratio` and `descents_ratio` were redundant (`desc = 1 - adj`).
- Action: removed `descents_ratio`.

- Finding: `value_range_norm` was saturated (many values near/at 1), low discrimination.
- Action: removed `value_range_norm` from `v1`.

- Finding: `length_norm` had low resolution because only 4 size levels were generated.
- Action: expanded synthetic default size grid to 10 levels in dataset generator.

- Finding: raw `skewness` and `kurtosis_excess` had extreme values and risked unstable model training.
- Action: replaced raw values with transformed features:
- `skewness_t = sign(skewness) * log1p(abs(skewness))`
- `kurtosis_excess_t = sign(kurtosis_excess) * log1p(abs(kurtosis_excess))`

- Current `v1` feature set:
- `length_norm`
- `adj_sorted_ratio`
- `duplicate_ratio`
- `dispersion_ratio`
- `runs_ratio`

- Current `v2` additions:
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

- Validation status target:
- no NaN/inf
- no duplicate `sample_id`
- bounded features in expected range
- deterministic rerun with same seed
