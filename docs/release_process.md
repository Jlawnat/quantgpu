# QuantGPU Release Process

## 1. Purpose

This document defines how QuantGPU versions, validates, tags, and publishes
public releases.

A release should identify a project state that is:

- numerically validated,
- reproducible,
- packageable,
- documented,
- and traceable to an exact Git commit.

---

## 2. Version format

QuantGPU uses semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Examples:

```text
0.1.0
0.2.0
0.2.1
1.0.0
```

During the pre-1.0 development phase:

- `MINOR` versions may introduce substantial new functionality,
- `PATCH` versions contain compatible fixes and small improvements.

The initial public release is:

```text
0.1.0
```

Git tags use the `v` prefix:

```text
v0.1.0
```

---

## 3. Version source

The canonical package version is defined in:

```text
src/quantgpu/__init__.py
```

Example:

```python
__version__ = "0.1.0"
```

`pyproject.toml` loads this value dynamically for package builds.

The Python package version and Git release tag must correspond.

For example:

```text
package version: 0.1.0
Git tag:         v0.1.0
```

---

## 4. Release candidate requirements

Before a revision can become a public release, the following must be true:

1. the working tree is clean,
2. the CPU quality gate passes,
3. required GPU validation has passed,
4. package builds succeed,
5. the built wheel installs in a clean environment,
6. the installed package reports the expected version,
7. release metadata is correct,
8. documentation describes the released implementation,
9. benchmark claims remain traceable to preserved evidence,
10. the changelog is updated for the release.

A release tag must not be created from an unvalidated development state.

---

## 5. CPU release gate

The normal release-quality CPU gate is:

```bash
ruff format --check src tests benchmarks
ruff check src tests benchmarks
mypy src
python -m compileall -q benchmarks
pytest -m "not gpu" --cov=quantgpu --cov-report=term-missing -q
git diff --check
```

The required CPU coverage floor is 95%.

GitHub Actions must also complete successfully for the release commit.

---

## 6. GPU release gate

GPU-dependent functionality is validated separately from normal hosted CI.

The canonical GPU reference environment uses an NVIDIA Tesla T4.

For a release containing GPU functionality, the documented GPU quality gate
must have passed for the relevant implementation.

A skipped GPU test does not count as successful GPU validation.

If GPU implementation code changes after validation, the affected GPU tests
must be rerun before release.

---

## 7. Package build gate

Before release, remove previous local build artifacts:

```bash
rm -rf build dist
```

Build the source distribution and wheel:

```bash
python -m build
```

Expected artifacts include:

```text
dist/quantgpu-<version>.tar.gz
dist/quantgpu-<version>-py3-none-any.whl
```

Both artifacts should correspond to the intended release version.

---

## 8. Clean-install verification

The built wheel must be tested independently of the development checkout.

A temporary environment should be created and the wheel installed into it.

The installed package must:

- import successfully,
- report the expected version,
- load from the temporary environment rather than the repository source tree,
- and successfully execute at least one representative public function.

This protects against packaging errors that normal editable development
installs may hide.

---

## 9. Release metadata

Before release, verify package metadata including:

- package name,
- version,
- summary,
- Python requirement,
- license,
- project URLs,
- keywords,
- classifiers.

The final package description is sourced from the root `README.md`.

Therefore the final package build must be performed after the release README
has been completed.

---

## 10. Changelog

The release entry in:

```text
CHANGELOG.md
```

must describe the major functionality, validation, reproducibility, and
performance work included in the release.

During development the release heading may use:

```text
## [0.1.0] - Unreleased
```

Immediately before the final release, replace `Unreleased` with the actual
release date.

For example:

```text
## [0.1.0] - 2026-09-06
```

---

## 11. Final release commit

The release tag should point to a clean commit containing:

- final source code,
- final tests,
- release documentation,
- changelog,
- license,
- package metadata,
- and final root README.

The release commit must pass GitHub Actions before the tag is created.

---

## 12. Tagging

QuantGPU uses annotated Git tags for public releases.

Example:

```bash
git tag -a v0.1.0 -m "QuantGPU v0.1.0"
```

Push the tag with:

```bash
git push origin v0.1.0
```

Do not create the tag until the release commit has passed all required
validation.

Published release tags should not be moved or rewritten.

---

## 13. GitHub Release

After the release tag is pushed, create a GitHub Release corresponding to the
same tag.

For the initial release:

```text
Tag:   v0.1.0
Title: QuantGPU v0.1.0
```

The release notes should summarize:

- supported CPU and GPU backends,
- numerical validation,
- benchmark methodology,
- headline performance results,
- reproducibility guarantees,
- canonical validation environments.

Detailed release information should remain consistent with `CHANGELOG.md` and
the root `README.md`.

---

## 14. Benchmark evidence

Creating a release does not require rewriting or overwriting historical
benchmark artifacts.

Existing validated benchmark CSVs should remain preserved.

Headline performance claims must remain traceable to:

- benchmark methodology,
- hardware environment,
- workload parameters,
- validation state,
- and the relevant Git revision.

If future implementation changes require new benchmark runs, new result
artifacts should be written under new versioned filenames.

---

## 15. Post-release changes

Development after a release continues from `main`.

Changes made after `v0.1.0` are not part of the `v0.1.0` release, even if they
are later present on `main`.

Bug fixes intended for a patch release should increment the patch version.

Example:

```text
0.1.0 → 0.1.1
```

Larger compatible feature additions normally increment the minor version.

Example:

```text
0.1.0 → 0.2.0
```

---

## 16. Release principle

QuantGPU follows the rule:

> A release is a validated and reproducible project state, not merely a version
> number.

The Git tag, package version, source code, documentation, validation evidence,
and benchmark provenance should tell the same story.