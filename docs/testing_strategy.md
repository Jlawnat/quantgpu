# QuantGPU Testing Strategy

## 1. Purpose

QuantGPU uses a layered testing strategy to verify:

- software correctness,
- quantitative correctness,
- statistical calibration,
- cross-backend consistency,
- reproducibility,
- GPU-specific behavior,
- and benchmark integrity.

Performance results are accepted only after the corresponding implementation
passes the relevant correctness checks.

The testing design deliberately separates CPU validation from GPU validation
because the development and CI environments do not provide the same hardware.

---

## 2. Test tiers

The test suite is organized into three primary tiers:

```text
tests/
├── unit/
├── numerical/
└── integration/
```

Pytest markers are applied automatically from the directory structure.

### Unit tests

Marker: `unit`

Purpose:

- isolated function behavior,
- input validation,
- deterministic edge cases,
- benchmark helper behavior,
- tolerance rules,
- RNG validation,
- result contracts.

These tests are intended to be fast and suitable for frequent local execution.

### Numerical tests

Marker: `numerical`

Purpose:

- analytical invariants,
- statistical calibration,
- Monte Carlo convergence,
- GBM moment checks,
- no-arbitrage relationships,
- deterministic limiting cases.

Examples include:

- Black-Scholes call and put bounds,
- monotonicity with respect to spot and strike,
- volatility monotonicity,
- scale homogeneity,
- GBM mean and variance checks,
- Monte Carlo standard-error calibration.

These tests verify quantitative behavior rather than only software mechanics.

### Integration tests

Marker: `integration`

Purpose:

- cross-component behavior,
- backend parity,
- RNG policy across implementations,
- analytical-reference agreement,
- CPU/GPU consistency.

Integration tests may be CPU-safe or GPU-dependent.

GPU-dependent integration tests also receive the `gpu` marker.

---

## 3. GPU test classification

Tests requiring NVIDIA CUDA hardware are marked `gpu`.

This includes validation of:

- PyTorch CUDA,
- reduced-intermediate FP32 CUDA,
- compiled CUDA,
- Triton CUDA,
- CPU versus CUDA parity,
- FP64 versus FP32 parity.

GPU tests are intentionally excluded from normal CPU CI.

This avoids treating a skipped CUDA test as equivalent to actual GPU
validation.

---

## 4. Canonical local commands

### Fast development loop

```bash
pytest -m "unit and not gpu" -q
```

### Full local CPU correctness gate

```bash
pytest -m "not gpu" -q
```

### CPU coverage gate

```bash
pytest -m "not gpu" \
  --cov=quantgpu \
  --cov-report=term-missing \
  -q
```

The current CPU-testable source baseline achieves 100% coverage.

The enforced minimum is 95%.

CUDA-specific implementation files are excluded from local CPU coverage
because they cannot be meaningfully exercised without CUDA hardware.

---

## 5. Static quality gates

QuantGPU uses automated static quality gates to keep the Python codebase
consistent and reviewable.

### Ruff formatting

Canonical formatting is enforced with:

```bash
ruff format --check src tests benchmarks
```

This checks that production code, tests, and benchmark scripts all conform to
the project formatter without modifying files.

### Ruff linting

Linting is enforced with:

```bash
ruff check src tests benchmarks
```

The configured lint rules cover:

- Python errors,
- import ordering,
- style issues,
- Python-version modernization,
- common bug patterns.

### Strict mypy

Static type checking is enforced with:

```bash
mypy src
```

Strict typing is applied across the normal Python implementation.

The Triton backend has narrowly scoped mypy exceptions for limitations in the
Triton JIT DSL. These exceptions do not disable type checking for the rest of
QuantGPU.

### Benchmark syntax validation

Benchmark entry points are syntax-checked with:

```bash
python -m compileall -q benchmarks
```

This ensures benchmark, experiment, and profiling scripts remain valid Python
even when they are not executed in the local CPU environment.

---

## 6. GitHub Actions CPU CI

The workflow is stored at:

```text
.github/workflows/ci.yml
```

It runs on pushes and pull requests targeting `main`.

The CPU quality gate performs:

1. repository checkout,
2. Python 3.12 setup,
3. CPU PyTorch installation,
4. development dependency installation,
5. Ruff formatting validation,
6. Ruff linting,
7. strict mypy,
8. benchmark syntax validation,
9. CPU pytest execution,
10. CPU coverage validation.

The CI coverage floor is 95%.

The CI test command excludes GPU-marked tests because standard hosted GitHub
Actions runners do not provide the canonical NVIDIA GPU environment.

GPU correctness is therefore validated separately using the documented Tesla T4
workflow.

---

## 7. Canonical GPU quality gate

GPU validation is performed on a Tesla T4 environment.

The clean validation workflow uses:

```text
/kaggle/working/quantgpu_repo
```

with:

```text
PYTHONPATH=/kaggle/working/quantgpu_repo/src
```

A fresh repository clone is used so validation corresponds directly to a
committed Git revision.

Canonical command:

```bash
PYTHONPATH=/kaggle/working/quantgpu_repo/src pytest -m gpu -q
```

Validated environment:

- PyTorch 2.10.0+cu128
- CUDA 12.8
- Tesla T4
- Triton 3.6.0

Validated GPU quality-gate result:

```text
14 passed
```

The number of deselected CPU tests may increase as the CPU test suite grows,
so only the GPU pass count is treated as the stable validation result.

PyTorch JIT deprecation warnings observed in this environment are warnings,
not correctness failures.

---

## 8. Numerical acceptance policy

A performance result is not accepted solely because the implementation
executes.

For Monte Carlo workloads, validation may include:

- analytical reference comparison,
- Monte Carlo tolerance checks,
- reported standard-error validation,
- reproducibility checks,
- convergence checks,
- backend parity.

Statistical comparisons use uncertainty-aware tolerances rather than requiring
bitwise equality between independent random-number implementations.

Deterministic same-backend runs with the same seed are expected to be
reproducible according to the backend RNG policy.

---

## 9. CPU/GPU parity

Different backends are not required to generate identical random paths.

Instead, pricing estimates must be statistically consistent within combined
Monte Carlo uncertainty.

This allows NumPy, PyTorch CPU, CUDA, compiled CUDA, and Triton backends to use
their native RNG implementations while still enforcing a common quantitative
correctness standard.

---

## 10. Performance regression testing

The performance regression policy is documented in:

```text
docs/performance_regression_policy.md
```

Performance is not enforced as a hard GitHub Actions gate because benchmark
timings are hardware-dependent.

For like-for-like Tesla T4 reruns:

- slowdown <= 10%: pass,
- slowdown > 10% and <= 20%: warning and rerun,
- slowdown > 20%: performance regression candidate.

A suspected regression must be reproduced before being treated as genuine.

---

## 11. Benchmark correctness gate

QuantGPU follows the rule:

> correctness before speed.

Every benchmark candidate must pass the required numerical validation before
its latency or throughput result is accepted.

A faster implementation that fails quantitative correctness is not considered
a valid optimization.

---

## 12. Testing architecture summary

```text
Local development
        │
        ├── unit tests
        ├── numerical tests
        ├── CPU integration tests
        ├── Ruff
        ├── strict mypy
        └── coverage
                │
                ▼
        GitHub Actions CPU Gate
                │
                ▼
        committed revision
                │
                ▼
        Tesla T4 GPU Gate
        ├── CUDA
        ├── FP32 CUDA
        ├── compiled CUDA
        ├── Triton
        └── cross-backend parity
                │
                ▼
        validated benchmark result
```

This structure ensures that QuantGPU does not confuse code execution,
statistical correctness, numerical validation, and hardware performance.
Each is tested explicitly at the appropriate layer.