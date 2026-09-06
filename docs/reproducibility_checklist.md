# QuantGPU Reproducibility Checklist

Use this checklist before accepting a benchmark, validation run, or release
artifact as reproducible.

## 1. Source provenance

- [ ] The exact Git commit SHA is known.
- [ ] The working-tree state is recorded.
- [ ] A dirty working tree is not presented as if it were the committed revision.
- [ ] The QuantGPU project version is recorded.
- [ ] The benchmark schema version is recorded.

## 2. Workload provenance

- [ ] Backend is recorded.
- [ ] Device is recorded.
- [ ] Dtype is recorded.
- [ ] Path count is recorded.
- [ ] Random seed is recorded.
- [ ] Warm-up count is recorded.
- [ ] Repetition count is recorded.
- [ ] Financial workload parameters match the documented benchmark definition.
- [ ] Any deviation from the canonical workload is explicitly documented.

## 3. Software environment

- [ ] Python version is recorded.
- [ ] NumPy version is recorded.
- [ ] PyTorch version is recorded.
- [ ] CUDA version is recorded where relevant.
- [ ] Triton version is recorded where relevant.
- [ ] The environment corresponds to an actual validated run rather than guessed versions.
- [ ] Important canonical environments have a committed reference snapshot.

## 4. Hardware environment

- [ ] CPU model is recorded where available.
- [ ] GPU model is recorded for GPU results.
- [ ] GPU performance comparisons use the same hardware class.
- [ ] Results from different hardware are not presented as direct like-for-like regressions.

## 5. Randomness

- [ ] Canonical benchmark seed policy is followed.
- [ ] Seeded same-backend reproducibility is verified where supported.
- [ ] `seed=None` is treated as intentionally non-reproducible.
- [ ] Cross-backend comparisons do not assume identical RNG streams.

## 6. Numerical correctness

- [ ] Analytical-reference checks pass where available.
- [ ] Financial invariants pass.
- [ ] Monte Carlo tolerance checks pass.
- [ ] Standard-error validation passes.
- [ ] Cross-backend parity passes.
- [ ] Reduced-precision backends satisfy their numerical acceptance policy.
- [ ] Performance results are not accepted when numerical validation fails.

## 7. CPU validation

- [ ] CPU validation is performed from a clean repository checkout.
- [ ] Ruff passes.
- [ ] Strict mypy passes.
- [ ] CPU pytest suite passes.
- [ ] Coverage meets or exceeds the configured threshold.
- [ ] Canonical CPU reference environment is recorded.

Current reference:

`environments/cpu_reference_2026-09.txt`

## 8. GPU validation

- [ ] GPU validation is performed from a clean repository checkout.
- [ ] CUDA is available.
- [ ] GPU model is confirmed.
- [ ] GPU pytest suite passes.
- [ ] Triton tests execute rather than being silently skipped.
- [ ] Canonical Tesla T4 reference environment is recorded.

Current reference:

`environments/tesla_t4_reference_2026-09.txt`

## 9. Benchmark integrity

- [ ] The timed workload still computes the required financial quantity.
- [ ] Required computation has not been removed solely to improve timing.
- [ ] CUDA timing includes the documented synchronization policy.
- [ ] Warm-up runs are excluded from reported statistics.
- [ ] Median latency is used as the primary timing statistic.
- [ ] Benchmark results are saved as artifacts rather than copied only from terminal output.

## 10. Performance comparison

- [ ] Compared results use the same workload.
- [ ] Compared results use the same path count.
- [ ] Dtype differences are explicitly identified.
- [ ] Hardware differences are explicitly identified.
- [ ] Timing methodology is unchanged.
- [ ] Numerical correctness passes before speedup is reported.
- [ ] Performance regression thresholds follow
      `docs/performance_regression_policy.md`.

## 11. Artifact policy

- [ ] Historical benchmark artifacts are preserved.
- [ ] New schema versions write to new versioned result files.
- [ ] Existing canonical result files are not silently rewritten.
- [ ] Headline performance numbers can be traced back to committed benchmark artifacts.

## 12. Reproduction acceptance

A reproduction is accepted only when:

1. the source revision is identifiable,
2. the workload is identifiable,
3. the relevant software and hardware environment is identifiable,
4. numerical validation passes,
5. and performance comparisons follow the documented methodology.

A matching runtime without matching provenance and numerical correctness is not
considered a successful reproduction.