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