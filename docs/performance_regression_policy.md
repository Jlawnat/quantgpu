# QuantGPU Performance Regression Policy

## Purpose

QuantGPU treats performance as a measured engineering property, but benchmark
results are only compared when the execution environments and workloads are
sufficiently equivalent.

Numerical correctness always takes precedence over performance.

A backend that fails the QuantGPU correctness gate is not eligible for
performance comparison.

## Canonical GPU workload

The current GPU optimization reference uses:

- European call Monte Carlo pricing under GBM
- spot = 100
- strike = 100
- maturity = 1 year
- risk-free rate = 5%
- volatility = 20%
- 10,000,000 paths
- seed = 42
- Tesla T4
- 3 warm-up runs
- 10 measured repetitions

The canonical optimization results are stored in:

`benchmarks/results/cuda_optimization_comparison_v2.csv`

## Like-for-like comparison requirement

Performance regression decisions require matching:

- backend
- workload
- path count
- dtype
- GPU model
- benchmark methodology
- timing definition
- correctness gate

Results from different GPU models or materially different software
environments must not be treated as direct performance regressions.

## Regression thresholds

For a like-for-like rerun against the frozen reference median:

- slowdown <= 10%: pass
- slowdown > 10% and <= 20%: warning; rerun benchmark
- slowdown > 20%: performance regression candidate

A performance regression candidate must be reproduced before being accepted as
a genuine regression.

## Confirmation procedure

For a suspected regression:

1. Confirm that the numerical correctness gate passes.
2. Confirm the workload and hardware match the reference benchmark.
3. Repeat the canonical benchmark in a fresh run.
4. Compare median end-to-end latency rather than a single timing.
5. Investigate the implementation or environment if the slowdown remains above
   20%.

## CI policy

CPU correctness, static analysis, and coverage are enforced automatically in
GitHub Actions.

GPU performance is not used as a hard GitHub CI gate because standard hosted
runners do not provide the canonical Tesla T4 environment.

GPU correctness and performance validation are performed separately on the
documented GPU environment.

## Optimization acceptance rule

A new optimization is accepted only when it:

1. passes numerical validation,
2. preserves the backend result contract,
3. demonstrates repeatable performance improvement under comparable conditions,
4. does not rely on a benchmark methodology change to create an artificial
   speedup.