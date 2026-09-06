# QuantGPU Repository Conventions

## 1. Purpose

QuantGPU uses explicit repository conventions so that quantitative logic,
hardware-specific implementations, tests, benchmarks, documentation, and
reproducibility artifacts remain clearly separated.

The repository should remain understandable to a new contributor without
requiring knowledge of the project's development history.

---

## 2. Top-level repository structure

The canonical repository layout is:

```text
quantgpu/
├── .github/
│   └── workflows/
├── benchmarks/
│   └── results/
├── docs/
├── environments/
├── src/
│   └── quantgpu/
├── tests/
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

Each category of project artifact has an explicit home.

---

## 3. Source package

Production Python code lives under:

```text
src/quantgpu/
```

The current package structure is:

```text
src/quantgpu/
├── backends/
├── benchmarking/
├── pricing/
├── simulation/
└── validation/
```

Responsibilities are separated as follows:

- `backends/` — CPU, CUDA, compiled, and Triton implementations
- `benchmarking/` — timing, metadata, provenance, schema, and benchmark validation
- `pricing/` — financial pricing logic and analytical reference models
- `simulation/` — stochastic simulation and random-number policy
- `validation/` — numerical tolerances and cross-backend parity

Reusable library logic belongs in `src/quantgpu/`, not inside benchmark scripts.

---

## 4. Benchmark scripts

Executable performance work lives under:

```text
benchmarks/
```

Naming conventions are:

```text
benchmark_*.py
experiment_*.py
profile_*.py
```

Their roles are distinct:

- `benchmark_*.py` — canonical or repeatable benchmark entry points
- `experiment_*.py` — exploratory optimization experiments
- `profile_*.py` — profiler-focused diagnostic scripts

Exploratory scripts must not silently replace canonical benchmark methodology.

---

## 5. Benchmark results

Validated benchmark artifacts live under:

```text
benchmarks/results/
```

Historical result files are preserved.

A new benchmark schema, methodology revision, or materially different
experiment should write to a new versioned artifact rather than overwrite
historical evidence.

Raw scratch output belongs under ignored locations such as:

```text
benchmarks/results/raw/
profiles/
```

Headline performance claims should be derived from preserved benchmark
artifacts rather than transient terminal output.

---

## 6. Tests

Tests live under:

```text
tests/
├── unit/
├── numerical/
└── integration/
```

Test filenames use:

```text
test_*.py
```

The directory structure determines the primary test tier.

GPU-dependent tests additionally use the `gpu` pytest marker.

Unknown pytest markers are rejected.

---

## 7. Documentation

Engineering documentation lives under:

```text
docs/
```

The root `README.md` is the public entry point to the project.

Detailed design, validation, benchmarking, reproducibility, workflow, and
release policies belong in focused documents under `docs/`.

Documentation should describe the current implementation rather than historical
intent.

---

## 8. Environment snapshots

Reference software environments live under:

```text
environments/
```

Environment snapshots document the software stack used for validated CPU and
GPU reproduction.

They are reproducibility evidence and should remain attributable to the
corresponding validated project state.

---

## 9. GitHub automation

Repository automation lives under:

```text
.github/
```

Continuous integration workflows live under:

```text
.github/workflows/
```

CI should validate the normal CPU-safe quality gate.

Hardware-specific GPU validation remains separate when hosted CI does not
provide the required hardware.

---

## 10. Python naming

Python modules, functions, and variables use:

```text
snake_case
```

Classes and protocols use:

```text
PascalCase
```

Constants use:

```text
UPPER_SNAKE_CASE
```

Names should describe quantitative or engineering meaning rather than temporary
implementation details.

---

## 11. Backend naming

Backend names should identify the execution strategy clearly.

Examples include:

```text
numpy_cpu
torch_cpu
torch_cuda
torch_cuda_fp32
torch_cuda_compiled_fp32
triton_cuda_fp32
```

Precision differences must be explicit when they materially affect numerical
or performance interpretation.

---

## 12. Type checking

Production Python code is expected to pass strict mypy validation.

Type-checking exceptions should be narrowly scoped to specific technical
limitations.

Project-wide disabling of strict typing is not acceptable solely to accommodate
one backend or external DSL.

---

## 13. Formatting and linting

Python code under:

```text
src/
tests/
benchmarks/
```

is formatted with Ruff.

Formatting validation:

```bash
ruff format --check src tests benchmarks
```

Lint validation:

```bash
ruff check src tests benchmarks
```

Formatting-only changes should be separated from functional changes when doing
so improves reviewability.

---

## 14. Docstrings

Public functions, classes, protocols, and important helpers should have concise
docstrings describing their purpose.

Docstrings should explain the contract rather than repeat the implementation
line by line.

Quantitative assumptions that materially affect interpretation should be made
explicit.

---

## 15. Errors and validation

Invalid user or workload inputs should fail explicitly.

Use `ValueError` for invalid argument values or quantitative domain violations.

Use `RuntimeError` when execution cannot proceed because a required runtime
capability or environment is unavailable.

Errors should fail close to the source of the invalid condition.

---

## 16. Quantitative code

Quantitative correctness takes priority over performance.

Optimization work must preserve the intended financial quantity and satisfy the
relevant numerical validation.

Precision changes, random-number behavior, timing boundaries, and statistical
tolerances must not be changed silently.

A faster implementation that fails quantitative validation is not accepted as a
valid optimization.

---

## 17. Canonical benchmarks

Canonical benchmark code must make the following explicit:

- workload parameters
- path count
- seed
- backend
- device
- dtype
- warm-up count
- repetition count
- timing methodology
- hardware metadata
- software metadata
- numerical validation status
- Git provenance

Benchmark methodology changes must be documented.

---

## 18. Commit messages

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

Commit messages should describe the resulting engineering change.

Avoid vague messages such as:

```text
update
changes
fix stuff
work
```

---

## 19. Main branch

`main` represents the current validated project state.

Changes committed to `main` should leave the relevant quality gates passing.

Larger, experimental, or multi-step changes may use feature branches.

Published history used for benchmark or release provenance should not be
rewritten merely for cosmetic cleanup.

---

## 20. Repository hygiene

Do not commit:

- virtual environments
- Python caches
- pytest, mypy, or Ruff caches
- local `.env` files
- credentials or API keys
- temporary logs
- raw profiler output
- unrelated generated artifacts

Validated benchmark result files and reference environment snapshots are
intentional tracked artifacts.

---

## 21. Repository principle

QuantGPU follows the rule:

> Source code, tests, benchmarks, documentation, and reproducibility evidence
> should each have an explicit home and an explicit contract.

A clean repository structure is part of the project's engineering and
reproducibility story.