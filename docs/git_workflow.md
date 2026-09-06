# QuantGPU Git Workflow

## 1. Purpose

QuantGPU uses a lightweight Git workflow designed to preserve a clean,
reviewable history without adding unnecessary process to a research-oriented
engineering project.

The Git history should make it possible to understand:

- what changed,
- why it changed,
- whether the change affected correctness or performance,
- and which validated project state produced a benchmark or release.

---

## 2. Main branch

`main` represents the current validated project state.

Changes committed to `main` should leave the relevant quality gates passing.

The normal CPU quality gate is:

```bash
ruff format --check src tests benchmarks
ruff check src tests benchmarks
mypy src
python -m compileall -q benchmarks
pytest -m "not gpu" --cov=quantgpu --cov-report=term-missing -q
```

GPU-dependent changes additionally require the documented GPU validation
workflow before they are treated as fully validated.

---

## 3. Branch policy

Small, isolated, well-tested changes may be committed directly to `main`
during single-developer work.

Feature branches are preferred when a change is:

- large,
- experimental,
- difficult to review as one step,
- likely to require multiple intermediate commits,
- or capable of temporarily breaking the validated main branch.

Recommended branch naming examples:

```text
feat/new-pricing-backend
perf/cuda-reduction
fix/seed-reproducibility
test/gpu-parity
docs/benchmark-methodology
```

Branch names should describe the engineering purpose of the work.

---

## 4. Commit scope

Each commit should represent one coherent engineering change.

Good commits include:

- adding one backend,
- adding one validation rule,
- introducing one benchmark schema change,
- adding one test family,
- updating one documented engineering policy,
- performing an explicitly isolated formatting-only change.

Avoid combining unrelated source, documentation, benchmark, and formatting
changes in the same commit when they can be reviewed independently.

---

## 5. Commit message format

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

Examples:

```text
feat: add benchmark source provenance
fix: enforce benchmark numerical validation
perf: add Triton CUDA backend
test: add cross-backend parity checks
docs: add reproducibility checklist
ci: enforce Ruff formatting
refactor: centralize software environment metadata
chore: normalize Python formatting
```

The subject should describe the resulting engineering change.

Avoid activity-only messages such as:

```text
update code
changes
work
fix stuff
try again
```

---

## 6. Functional versus formatting changes

Formatting-only changes should be isolated from functional changes whenever
the diff would otherwise become difficult to review.

A formatting commit must not intentionally alter behavior.

Example:

```text
chore: normalize Python formatting
```

This makes later code review and `git blame` interpretation clearer.

---

## 7. Benchmark-related commits

Benchmark code changes require additional care because they can change the
meaning of performance results.

A commit that changes any of the following should be treated as
benchmark-affecting:

- workload definition,
- path count,
- dtype,
- timing region,
- synchronization behavior,
- warm-up count,
- repetition count,
- backend implementation,
- correctness gate,
- benchmark schema.

Benchmark-affecting commits must not silently overwrite historical benchmark
artifacts.

New benchmark results should remain traceable to the exact commit that
generated them.

---

## 8. GPU changes

Changes affecting CUDA, compiled PyTorch, or Triton code should not be treated
as fully validated solely because CPU CI passes.

The expected workflow is:

1. pass the normal CPU quality gate,
2. commit the candidate revision,
3. validate that exact revision in the canonical GPU environment,
4. record or update GPU evidence only after the GPU gate passes.

A skipped GPU test is not equivalent to a GPU pass.

---

## 9. Documentation changes

Documentation should describe the current committed implementation.

When a code change alters:

- benchmark methodology,
- test commands,
- reproducibility policy,
- supported backend behavior,
- performance interpretation,

the corresponding documentation should be updated as part of the same logical
change or an immediately following documentation commit.

---

## 10. Pre-commit verification

Before committing a normal engineering change, check:

```bash
git diff --check
git status --short
```

Then run the relevant quality gates.

Before committing benchmark or GPU-related work, also verify that no accidental
result artifacts, profiler traces, or temporary files were generated.

---

## 11. Push verification

After pushing an important checkpoint:

```bash
git status --short
git log -1 --oneline
```

The local working tree should normally be clean.

GitHub Actions should also be allowed to complete successfully before the
revision is treated as the current validated CPU state.

---

## 12. History preservation

Published benchmark and release history should remain understandable.

Avoid rewriting shared history solely to make the commit graph cosmetically
cleaner.

Once a commit has been used as the provenance of a published benchmark,
environment snapshot, or release, its identity should be preserved.

---

## 13. Secret handling

Credentials, tokens, API keys, and private configuration must never be
committed.

If a secret is accidentally committed, deleting it in a later commit is not
sufficient. The credential must be treated as exposed and rotated.

---

## 14. Git principle

QuantGPU follows the rule:

> Every important project state should be identifiable, reviewable, and
> reproducible from Git history.

Git is not only source backup; it is part of the project's benchmark,
validation, and reproducibility system.