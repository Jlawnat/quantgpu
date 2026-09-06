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
├── benchmarks/
├── docs/
├── environments/
├── src/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md