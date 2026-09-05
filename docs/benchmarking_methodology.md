# QuantGPU Benchmarking Methodology

This document defines the benchmark standard used throughout QuantGPU.

The goal is to make performance results reproducible, comparable, and
numerically meaningful across CPU and GPU implementations.

## 1. Canonical Workload

The baseline workload is Monte Carlo pricing of a European call option under
geometric Brownian motion.

Unless an experiment explicitly states otherwise, benchmarks use:

- Spot price: 100
- Strike price: 100
- Maturity: 1 year
- Risk-free rate: 5%
- Volatility: 20%
- Random seed: 42

The standard path-count ladder is:

- 10,000
- 100,000
- 1,000,000
- 5,000,000
- 10,000,000

Changing any canonical workload parameter must be reported explicitly.

## 2. Numerical Correctness Gate

Performance results are meaningful only for numerically valid implementations.

Before a backend is used for headline benchmarking, it must pass the relevant:

- analytical reference tests,
- pricing invariant tests,
- Monte Carlo statistical validation,
- reproducibility tests,
- precision-aware tolerance checks,
- cross-backend consistency tests.

A faster implementation that fails numerical validation is not considered a
valid performance improvement.

## 3. CPU Timing

CPU benchmarks use end-to-end wall-clock timing.

The timed region includes the complete pricing call, including:

- random-number generation,
- path simulation,
- payoff calculation,
- discounting,
- reduction,
- result construction.

The standard CPU timing configuration is:

- Warm-up runs: 1
- Measured repetitions: 5
- Primary statistic: median elapsed time

Minimum and maximum measured times are also recorded for diagnostic purposes.

## 4. CUDA Timing

CUDA benchmarks report two timing measurements.

### Device timing

Device timing measures GPU execution using CUDA-aware timing with explicit
synchronization.

The standard CUDA timing configuration is:

- Warm-up runs: 3
- Measured repetitions: 10
- Primary statistic: median elapsed time

### End-to-end timing

End-to-end timing measures the complete host-observed pricing call using wall
clock time.

CUDA is synchronized before timing interpretation so asynchronous execution is
not mistaken for completed work.

Headline CPU-versus-GPU speedups use end-to-end median latency.

## 5. Warm-up Policy

Warm-up executions are never included in reported benchmark statistics.

Warm-up runs allow runtime initialization, CUDA context initialization, memory
allocation behaviour, and framework setup costs to stabilize before
measurement.

Experiments involving compilation may use additional warm-up runs, but this
must be stated explicitly.

## 6. Reported Statistics

Median latency is the primary benchmark statistic.

Median is preferred to a single execution because it is less sensitive to
transient system noise and occasional outliers.

Each benchmark records, where applicable:

- median device latency,
- median end-to-end latency,
- minimum latency,
- maximum latency,
- throughput,
- speedup versus NumPy CPU,
- speedup versus PyTorch CPU.

## 7. Throughput

Throughput is reported as:

\[
\text{throughput}
=
\frac{\text{number of Monte Carlo paths}}
{\text{elapsed seconds}}
\]

For CUDA measurements, throughput is based on device execution time unless
explicitly labelled otherwise.

## 8. Speedup

Headline backend speedup is calculated using end-to-end median latency.

For backend \(B\),

\[
\text{speedup versus NumPy}
=
\frac{T_{\text{NumPy CPU}}}
{T_B}
\]

and

\[
\text{speedup versus PyTorch CPU}
=
\frac{T_{\text{PyTorch CPU}}}
{T_B}.
\]

Speedups must compare identical financial workloads and path counts.

## 9. Precision Policy

FP64 is the default numerical baseline.

FP32 results must always be labelled explicitly.

Performance comparisons between FP32 and FP64 are treated as precision
experiments rather than interchangeable benchmark results.

Any reduced-precision implementation must satisfy the corresponding numerical
validation policy before its performance is reported.

## 10. Randomness and Reproducibility

Canonical benchmark runs use seed 42.

A fixed seed ensures that repeated benchmark runs use a reproducible stochastic
workload within each backend.

Different numerical libraries and devices are not required to generate
identical random streams.

Cross-backend correctness is therefore evaluated statistically rather than by
requiring pathwise equality.

## 11. System Metadata

Benchmark records must include sufficient system information to interpret the
result.

This includes, where available:

- backend,
- device,
- dtype,
- path count,
- Python version,
- PyTorch version,
- CUDA version,
- GPU model,
- CPU model,
- operating system,
- warm-up count,
- repetition count,
- random seed,
- benchmark schema version.

Performance numbers should not be presented without their hardware and
precision context.

## 12. Experimental Benchmarks

Profiler runs, fused implementations, FP32 experiments, compilation
experiments, and future custom CUDA kernels may use specialized methodology.

Such experiments must clearly identify any departure from the canonical
benchmark standard.

Experimental results must not silently replace the canonical baseline.

## 13. Benchmark Integrity

Benchmark code must not remove required financial computation solely to improve
timing results.

Optimizations may change implementation strategy, memory layout, operation
fusion, precision, compilation, or execution backend, but the financial
quantity being computed must remain equivalent under the QuantGPU validation
standard.