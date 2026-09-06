# Changelog

All notable changes to QuantGPU are documented in this file.

The project follows semantic versioning for public releases.

## [0.1.0] - Unreleased

Initial public release of QuantGPU.

### Added

- NumPy CPU Monte Carlo backend.
- PyTorch CPU Monte Carlo backend.
- PyTorch CUDA backend.
- Fused PyTorch CUDA backend.
- Compiled PyTorch CUDA backend.
- Triton CUDA backend.
- Common backend protocol.
- European call pricing under geometric Brownian motion.
- Black-Scholes analytical reference pricing.
- Deterministic random-number-generation policy.
- Cross-backend numerical parity validation.
- Statistical Monte Carlo convergence checks.
- Quantitative invariant tests.
- CPU and GPU benchmark infrastructure.
- CUDA timing utilities with explicit synchronization.
- Benchmark schema and provenance metadata.
- Git commit and working-tree provenance capture.
- Software and hardware environment metadata capture.
- Canonical CPU and Tesla T4 environment snapshots.
- Performance regression policy.
- Reproducibility standard and checklist.
- Repository conventions and Git workflow documentation.
- Contributor workflow documentation.
- GitHub Actions CPU quality gate.
- Ruff formatting and lint enforcement.
- Strict mypy validation.
- Pytest marker enforcement.
- Minimum 95% CPU coverage requirement.

### Performance

Canonical Tesla T4 European-call Monte Carlo workload at 10 million paths:

- NumPy CPU: approximately 341 ms.
- PyTorch CPU: approximately 571 ms.
- PyTorch CUDA FP64: approximately 10.7 ms end-to-end.
- PyTorch CUDA FP32: approximately 3.08 ms.
- Selected Triton CUDA FP32 kernel: approximately 0.683 ms device time.
- Triton throughput: approximately 14.8 billion paths per second.
- Triton CUDA FP32 achieved approximately 2.1× speedup over compiled PyTorch FP32.
- Triton CUDA FP32 achieved approximately 4.5× speedup over eager PyTorch FP32.

Performance figures are tied to the documented benchmark methodology and
canonical Tesla T4 environment. Historical benchmark artifacts are preserved
under `benchmarks/results/`.

### Validation

- CPU quality gate validated with full production-code coverage above the
  required 95% threshold.
- Canonical Tesla T4 GPU quality gate validated with 14 GPU tests passing.
- Benchmark outputs require explicit numerical validation before being accepted.
- CUDA benchmark rows require GPU provenance metadata.
- Historical benchmark results remain immutable and versioned.

### Reproducibility

Version 0.1.0 establishes the project's reproducibility contract:

- benchmark schema versioning,
- Git provenance,
- clean/dirty working-tree state,
- workload parameter capture,
- software version capture,
- hardware metadata capture,
- canonical environment snapshots,
- validation status,
- preserved benchmark artifacts.

A benchmark result should not be treated as a headline project result unless
its implementation, workload, environment, validation state, and Git revision
are identifiable.

---

## Release policy

Public releases use semantic versions:

```text
MAJOR.MINOR.PATCH
```

For example:

```text
0.1.0
```

The Git tag corresponding to a release uses the `v` prefix:

```text
v0.1.0
```

Release tags should only be created from a clean, validated commit after the
release documentation and repository presentation are complete.