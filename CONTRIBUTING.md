# Contributing to QuantGPU

QuantGPU is a quantitative finance engineering project focused on numerical
correctness, reproducible benchmarking, and CPU/GPU performance.

Contributions should preserve those principles.

## Development setup

QuantGPU requires Python 3.12 or later.

Create and activate a virtual environment, then install CPU PyTorch and the
development dependencies.

Example:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

GPU dependencies should be installed only in an appropriate CUDA-capable
environment.

## Local quality gate

Before committing a normal change, run:

```bash
ruff format --check src tests benchmarks
ruff check src tests benchmarks
mypy src
python -m compileall -q benchmarks
pytest -m "not gpu" --cov=quantgpu --cov-report=term-missing -q
git diff --check
```

The required CPU coverage floor is 95%.

## Test structure

Tests are organized into:

```text
tests/unit/
tests/integration/
tests/numerical/
```

GPU-dependent tests use the `gpu` pytest marker.

The main marker groups are:

```text
unit
integration
numerical
gpu
slow
```

Unknown markers are rejected by pytest.

## GPU validation

GPU correctness is validated separately from normal CPU CI.

The canonical GPU reference environment uses an NVIDIA Tesla T4.

A GPU-dependent change should not be considered fully validated until the
documented GPU quality gate passes in the canonical environment.

A skipped GPU test is not equivalent to a GPU pass.

## Benchmark discipline

Benchmark changes must preserve the documented benchmark methodology.

Do not change workload parameters, timing boundaries, synchronization,
warm-up counts, repetition counts, dtype, or validation rules without
explicitly documenting the change.

Historical benchmark result files must not be silently overwritten.

New benchmark artifacts should remain traceable to the exact Git commit that
produced them.

## Numerical correctness

Performance improvements must not weaken numerical correctness.

Changes affecting pricing, simulation, random number generation, backend
implementations, or numerical tolerances should include appropriate tests.

When a trusted analytical reference exists, benchmark outputs should be
validated against it before performance results are accepted.

## Code quality

Production Python code is expected to remain:

- formatted with Ruff,
- lint-clean,
- strictly typed with mypy,
- covered by tests,
- and explicit about errors and numerical assumptions.

Triton-specific type-checking exceptions should remain narrowly scoped.

## Git workflow

The project Git workflow is documented in:

```text
docs/git_workflow.md
```

Preferred commit prefixes are:

```text
feat:
fix:
perf:
test:
docs:
ci:
refactor:
chore:
```

Commits should represent coherent engineering changes.

Formatting-only changes should remain separate from functional changes when
possible.

## Repository hygiene

Do not commit:

- virtual environments,
- Python caches,
- test or type-checker caches,
- temporary profiling outputs,
- local `.env` files,
- API keys,
- credentials,
- temporary logs,
- unrelated generated artifacts.

Published benchmark artifacts under `benchmarks/results/` are intentional
project evidence and should be preserved.

## Documentation

When implementation changes affect benchmark methodology, validation,
reproducibility, supported backends, testing, or performance interpretation,
update the corresponding documentation.

Documentation should describe the current committed implementation.

## Pull requests

For larger or experimental changes, prefer a feature branch and pull request.

A pull request should make clear:

- what changed,
- why it changed,
- how it was validated,
- whether numerical behavior changed,
- whether benchmark results changed,
- and whether GPU validation is required.

## Contribution principle

QuantGPU follows one central rule:

> Correctness first, reproducibility second, performance third.

A faster implementation is only useful when its numerical behavior and
benchmark provenance remain trustworthy.