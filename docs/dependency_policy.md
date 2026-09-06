# QuantGPU Dependency Policy

## 1. Purpose

QuantGPU distinguishes between:

1. supported dependency ranges used to install and develop the project, and
2. exact environment snapshots used to reproduce specific benchmark results.

These serve different purposes and must not be treated as interchangeable.

---

## 2. Supported dependency ranges

Supported project dependencies are defined in:

`pyproject.toml`

Examples include:

- Python >= 3.12
- NumPy >= 2.0
- PyTorch >= 2.0
- supported development-tool ranges

These ranges describe versions that QuantGPU is intended to support.

They are not intended to reconstruct the exact environment of a historical
benchmark run.

---

## 3. Exact benchmark environments

Canonical benchmark results must record the exact software versions used for
the run.

Benchmark schema 1.1 records:

- Python version
- QuantGPU version
- NumPy version
- PyTorch version
- CUDA version
- Triton version
- Git commit
- Git working-tree state

These values describe the environment that actually produced the result.

---

## 4. Environment snapshots

Exact environment snapshots may be stored under:

`environments/`

Snapshots are reference artifacts for important reproducibility runs.

They are not the project's primary dependency specification.

A snapshot should correspond to a documented environment such as:

- canonical CPU validation,
- canonical Tesla T4 GPU validation,
- release validation.

Snapshots should be generated from the actual validated environment rather
than manually guessed.

---

## 5. Why QuantGPU does not pin all dependencies in pyproject.toml

Hard-pinning every dependency in the project metadata would make QuantGPU
unnecessarily tied to one machine or benchmark session.

For example, a canonical Tesla T4 benchmark may use one specific PyTorch and
CUDA stack, while the core CPU package remains valid on other supported
versions.

Therefore:

`pyproject.toml`

defines supported compatibility ranges, while:

`environments/`

records exact reference environments.

---

## 6. Reproducing a historical benchmark

To reproduce a benchmark as closely as possible:

1. check out the recorded Git commit,
2. confirm the benchmark schema and workload,
3. use the recorded software versions,
4. use the documented hardware class,
5. use the recorded random seed,
6. run the same benchmark methodology,
7. confirm numerical correctness before comparing performance.

The benchmark row itself remains the primary provenance record.

---

## 7. Snapshot update policy

Environment snapshots are immutable reference points once associated with a
canonical benchmark or release.

A newer environment should create a new snapshot rather than silently
overwriting the old one.

Examples:

`cpu_reference_2026-09.txt`

`tesla_t4_reference_2026-09.txt`

Future snapshots may use different names when tied to a release.

---

## 8. Installation versus reproduction

A normal installation asks:

> Which dependency versions are supported?

A reproduction asks:

> Which exact dependency versions produced this result?

QuantGPU intentionally answers these questions using different artifacts.

---

## 9. Dependency principle

> Supported ranges define compatibility; exact snapshots define provenance.

This separation allows QuantGPU to remain maintainable while preserving
benchmark reproducibility.