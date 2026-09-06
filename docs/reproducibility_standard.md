# QuantGPU Reproducibility Standard

## 1. Purpose

QuantGPU treats reproducibility as a first-class engineering requirement.

A result should be traceable to:

- a specific version of the source code,
- a documented quantitative workload,
- an explicit random seed where deterministic replay is required,
- the relevant software environment,
- the relevant hardware environment,
- and the benchmark or validation methodology used to generate it.

Reproducibility does not mean that every backend, device, or machine must
produce bitwise-identical floating-point values or identical benchmark
latencies.

Instead, QuantGPU distinguishes several forms of reproducibility.

---

## 2. Deterministic same-backend reproducibility

For a deterministic replay using the same:

- backend,
- seed,
- dtype,
- workload parameters,
- compatible software environment,
- and hardware class where relevant,

the numerical result should reproduce according to the backend's documented
RNG and execution behavior.

QuantGPU currently requires seeded repeated runs of the same backend to
reproduce the same `PricingResult` where this behavior is supported and tested.

A seed of `None` represents a fresh random stream and is therefore explicitly
not reproducible by design.

---

## 3. Cross-backend reproducibility

Different numerical libraries and devices are not required to generate the
same random-number stream.

For example, NumPy CPU and PyTorch CUDA may generate different simulated paths
for the same integer seed.

Cross-backend reproducibility therefore means that independently generated
results remain statistically consistent with:

- the analytical reference,
- Monte Carlo uncertainty,
- and the QuantGPU parity tolerance policy.

Bitwise equality across backends is not required.

---

## 4. Numerical reproducibility

A reproduced quantitative result must continue to satisfy the same numerical
correctness requirements as the original result.

Depending on the workload, these may include:

- analytical-reference agreement,
- financial invariants,
- Monte Carlo tolerance,
- convergence behavior,
- standard-error calibration,
- precision-aware tolerances,
- and cross-backend parity.

A numerically invalid result is not considered a successful reproduction even
if its runtime matches the original benchmark.

---

## 5. Performance reproducibility

Exact benchmark latency is not expected to reproduce across different
machines, GPU models, software stacks, or shared compute sessions.

A performance result is considered comparable only when the important
benchmark conditions are sufficiently equivalent.

These include:

- workload,
- backend,
- dtype,
- path count,
- timing methodology,
- warm-up count,
- repetition count,
- hardware model,
- and relevant software versions.

Median latency is the primary comparison statistic.

Performance comparisons must follow the QuantGPU performance regression
policy.

---

## 6. Source-code provenance

Every canonical benchmark result should be attributable to the exact QuantGPU
source revision that generated it.

The preferred source identifier is the Git commit SHA.

Benchmark provenance should therefore record, where available:

- Git commit SHA,
- project version,
- benchmark schema version.

A benchmark generated from a dirty working tree must not silently be presented
as if it came from the committed revision without recording that state.

---

## 7. Software environment provenance

Canonical results should record the software versions required to interpret or
reproduce the run.

Depending on the backend, this includes:

- Python,
- NumPy,
- PyTorch,
- CUDA runtime,
- Triton,
- and QuantGPU project version.

Exact environment reconstruction and normal package installation are separate
concerns.

The project dependency specification defines supported versions, while
canonical benchmark records or environment snapshots identify the versions
actually used for a reported result.

---

## 8. Hardware provenance

Performance results must include hardware context.

CPU results should identify the CPU model where available.

GPU results should identify:

- GPU model,
- CUDA availability,
- and relevant CUDA software version.

Results obtained on different hardware may both be valid, but they must not be
treated as directly interchangeable performance measurements.

---

## 9. Workload provenance

A reproducible benchmark must identify the workload that was executed.

For the current European-call Monte Carlo workload this includes:

- spot,
- strike,
- maturity,
- risk-free rate,
- volatility,
- path count,
- seed,
- dtype,
- backend.

The canonical benchmark workload is defined in
`docs/benchmarking_methodology.md`.

Any deviation from the canonical workload must be reported explicitly.

---

## 10. Benchmark artifact policy

Canonical benchmark results are stored under:

`benchmarks/results/`

Saved benchmark artifacts are the source of truth for reported performance
numbers.

Headline results should be derived from committed benchmark artifacts rather
than copied from transient terminal output.

Benchmark artifacts should retain enough metadata to identify:

1. what was run,
2. which code generated it,
3. where it ran,
4. which software stack was used,
5. how it was timed,
6. and whether it passed numerical validation.

---

## 11. Clean-room reproduction

A strong reproduction should be possible from a fresh repository checkout
rather than depending on uncommitted local files or notebook state.

CPU reproduction should be possible from a clean clone using the documented
Python environment and canonical commands.

GPU reproduction should use a clean clone in the documented CUDA environment.

The current canonical GPU validation workflow uses:

`/kaggle/working/quantgpu_repo`

with:

`PYTHONPATH=/kaggle/working/quantgpu_repo/src`

This prevents accidental dependence on stale notebook files or editable
installation state.

---

## 12. Reproducibility hierarchy

QuantGPU uses the following hierarchy:

### Level 1 — Code reproducibility

The exact source revision and workload can be identified.

### Level 2 — Numerical reproducibility

The result satisfies the same quantitative correctness standard.

### Level 3 — Same-backend seeded reproducibility

Repeated seeded execution reproduces the backend result where deterministic
replay is supported.

### Level 4 — Environment reproducibility

The important software and hardware environment can be reconstructed or
identified.

### Level 5 — Performance reproducibility

A like-for-like environment produces performance consistent with the
documented benchmark range and regression policy.

A result should not be described simply as "reproducible" without considering
which level is relevant.

---

## 13. Reproducibility principle

QuantGPU follows the rule:

> A benchmark number without provenance is not a reproducible benchmark.

Performance, numerical correctness, random-state control, source revision, and
environment metadata are treated as separate but connected parts of the same
reproducibility standard.