# QuantGPU

**High-performance quantitative finance research engine for reproducible CPU/GPU benchmarking, numerical validation, and custom GPU optimization.**

QuantGPU explores how a quantitative pricing workload moves from a conventional
CPU implementation to increasingly optimized GPU implementations while keeping
numerical correctness, benchmark methodology, and reproducibility explicit.

The current workload is Monte Carlo pricing of a European call option under
geometric Brownian motion, implemented across NumPy, PyTorch, CUDA,
`torch.compile`, and Triton.

The project follows one rule throughout:

> **Correctness first, reproducibility second, performance third.**

A faster implementation is not accepted unless it also passes the relevant
quantitative validation.

---

## Highlights

- NumPy and PyTorch CPU pricing backends
- PyTorch CUDA FP64 and FP32 implementations
- reduced-intermediate / fused CUDA implementation
- `torch.compile` optimized CUDA implementation
- custom Triton FP32 backend
- shared backend pricing contract
- Black-Scholes analytical validation
- Monte Carlo convergence and uncertainty checks
- cross-backend statistical parity testing
- reproducible benchmark schema and Git provenance
- CPU and GPU environment snapshots
- GitHub Actions CPU quality gate
- strict type checking, linting, formatting, and coverage enforcement
- clean wheel build and installation validation
- versioned benchmark artifacts and release process

Current package version:

```text
0.1.0
```

---

## Headline performance

Canonical workload:

```text
European call option
Spot              100
Strike            100
Maturity          1 year
Risk-free rate    5%
Volatility        20%
Monte Carlo paths 10,000,000
Seed              42
GPU               NVIDIA Tesla T4
```

All reported implementations passed the applicable numerical correctness gate.

### CPU to CUDA baseline

| Backend | Precision | Median end-to-end latency | Relative performance |
|---|---:|---:|---:|
| NumPy CPU | FP64 | 367.65 ms | baseline |
| PyTorch CPU | FP64 | 639.39 ms | baseline |
| PyTorch CUDA | FP64 | 10.67 ms | 34.45× vs NumPy CPU |
| PyTorch CUDA | FP64 | 10.67 ms | 59.91× vs PyTorch CPU |

The FP64 CUDA implementation already reduces the 10-million-path workload from
hundreds of milliseconds on CPU to approximately 10.7 ms on the Tesla T4.

### GPU optimization progression

| Backend | Precision | Median end-to-end latency | Optimization result |
|---|---:|---:|---:|
| PyTorch CUDA eager | FP32 | 3.060 ms | FP32 baseline |
| Reduced-intermediate CUDA | FP32 | 3.063 ms | no meaningful gain |
| `torch.compile` CUDA | FP32 | 1.429 ms | 2.14× vs eager FP32 |
| Triton CUDA | FP32 | **0.683 ms** | **4.48× vs eager FP32** |

The selected optimized backend is:

```text
triton_cuda_fp32
```

Its measured device latency is approximately:

```text
0.674 ms
```

corresponding to approximately:

```text
14.83 billion Monte Carlo paths / second
```

on the canonical Tesla T4 benchmark.

Within the same optimization benchmark, Triton is approximately:

- **4.48× faster** than eager PyTorch FP32
- **2.09× faster** than compiled PyTorch FP32
- **16.63× faster** than the FP64 CUDA implementation

The saved benchmark artifacts are the source of truth for these numbers.

See:

- [GPU optimization summary](docs/gpu_optimization_summary.md)
- [Benchmarking methodology](docs/benchmarking_methodology.md)
- [Benchmark results](benchmarks/results/)

---

## Optimization path

The project deliberately progresses through several implementation strategies
rather than jumping directly to a custom kernel.

```text
NumPy CPU
    │
    ▼
PyTorch CPU
    │
    ▼
PyTorch CUDA FP64
    │
    ▼
PyTorch CUDA FP32
    │
    ▼
Reduced-intermediate FP32
    │
    ▼
torch.compile FP32
    │
    ▼
Custom Triton FP32
```

This makes the performance gains attributable to specific engineering changes
rather than to a single opaque implementation.

The final Triton kernel fuses:

- GBM terminal-price transformation
- option payoff calculation
- discounting
- block payoff reduction
- block squared-payoff reduction

PyTorch remains responsible for Gaussian random-number generation.

The design reduces intermediate tensor materialization while preserving the
common QuantGPU pricing-result contract.

---

## Architecture

```text
                    Quantitative workload
                            │
                            ▼
                 EuropeanCallBackend protocol
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
      NumPy CPU        PyTorch CPU       PyTorch CUDA
                                               │
                         ┌─────────────────────┼──────────────────────┐
                         │                     │                      │
                         ▼                     ▼                      ▼
                      FP32                torch.compile             Triton
                         │                     │                      │
                         └─────────────────────┴──────────────────────┘
                                               │
                                               ▼
                                         PricingResult
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                    Numerical             Benchmark             Provenance
                    validation             timing                metadata
                         │                     │                     │
                         └─────────────────────┴─────────────────────┘
                                               │
                                               ▼
                                    Versioned benchmark artifact
```

The core package is separated into:

```text
src/quantgpu/
├── backends/       CPU, CUDA, compiled, and Triton implementations
├── benchmarking/   timing, metadata, schema, provenance, validation
├── pricing/        Black-Scholes and Monte Carlo pricing
├── simulation/     GBM simulation and RNG policy
└── validation/     parity and numerical tolerance rules
```

---

## Numerical correctness

QuantGPU does not treat successful execution as sufficient evidence that an
implementation is correct.

Validation includes:

- Black-Scholes analytical reference comparison
- call and put pricing invariants
- no-arbitrage bounds
- spot and strike monotonicity
- volatility monotonicity
- scale homogeneity
- GBM moment checks
- Monte Carlo convergence
- standard-error calibration
- seeded reproducibility
- cross-backend statistical parity
- precision-aware FP32 / FP64 tolerances

Different numerical libraries are not required to generate identical random
paths.

Instead, independent backend estimates must remain statistically consistent
with the analytical reference and Monte Carlo uncertainty.

A benchmark candidate that fails numerical validation is not accepted as a
performance improvement.

---

## Testing and quality gates

The test suite is divided into:

```text
tests/
├── unit/
├── numerical/
└── integration/
```

Pytest markers distinguish:

```text
unit
integration
numerical
gpu
slow
```

### CPU quality gate

```bash
ruff format --check src tests benchmarks
ruff check src tests benchmarks
mypy src
python -m compileall -q benchmarks
pytest -m "not gpu" --cov=quantgpu --cov-report=term-missing -q
```

The enforced CPU coverage floor is:

```text
95%
```

The current CPU-testable source baseline achieves:

```text
100% coverage
```

### GPU quality gate

GPU validation is performed separately in the canonical Tesla T4 environment.

Validated result:

```text
14 GPU tests passed
```

A skipped GPU test is not treated as equivalent to successful GPU validation.

See [Testing strategy](docs/testing_strategy.md).

---

## Reproducible benchmarking

QuantGPU records benchmark context rather than storing latency alone.

Canonical benchmark metadata includes:

- Git commit
- Git working-tree state
- QuantGPU version
- benchmark schema version
- backend
- device
- dtype
- workload parameters
- path count
- random seed
- warm-up count
- repetition count
- Python version
- NumPy version
- PyTorch version
- CUDA version
- Triton version
- CPU model
- GPU model
- validation status

Historical benchmark files are preserved instead of being silently overwritten.

New schema or methodology revisions write to new versioned artifacts.

See:

- [Reproducibility standard](docs/reproducibility_standard.md)
- [Reproducibility checklist](docs/reproducibility_checklist.md)
- [Performance regression policy](docs/performance_regression_policy.md)

---

## Canonical environments

Two reference environments are preserved under:

```text
environments/
```

### CPU reference

The CPU environment snapshot records the validated software stack used for
clean-room CPU reproduction.

### GPU reference

The canonical GPU environment uses:

```text
NVIDIA Tesla T4
Python 3.12.13
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
```

See [environments/README.md](environments/README.md).

---

## Installation

QuantGPU requires Python 3.12 or later.

### Core NumPy installation

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

### PyTorch and development installation

```bash
python -m pip install -e ".[torch,dev]"
```

For CUDA execution, install a PyTorch build compatible with the target CUDA
environment before running the GPU backends.

The Triton backend additionally requires a compatible Triton installation and
NVIDIA CUDA-capable hardware.

---

## Quick start

Run a NumPy CPU Monte Carlo price:

```python
from quantgpu.backends.numpy_cpu import price_european_call_numpy_cpu

result = price_european_call_numpy_cpu(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    n_paths=1_000_000,
    seed=42,
)

print(result.price)
print(result.standard_error)
```

Calculate the analytical reference:

```python
from quantgpu.pricing.black_scholes import black_scholes_call

reference = black_scholes_call(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
)

print(reference)
```

The Black-Scholes reference value for the canonical workload is approximately:

```text
10.4506
```

---

## Running benchmarks

CPU and GPU benchmark entry points live under:

```text
benchmarks/
```

Examples:

```bash
python benchmarks/benchmark_monte_carlo.py
python benchmarks/benchmark_backends.py
python benchmarks/benchmark_cpu_gpu.py
python benchmarks/benchmark_cuda_optimizations.py
```

GPU benchmarks require a compatible CUDA environment.

Profiling and experimental scripts are separated from the canonical benchmark
artifacts so exploratory work does not silently replace the project baseline.

---

## Repository structure

```text
quantgpu/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── benchmarks/
│   ├── results/
│   ├── benchmark_monte_carlo.py
│   ├── benchmark_backends.py
│   ├── benchmark_cpu_gpu.py
│   ├── benchmark_cuda_optimizations.py
│   ├── experiment_*.py
│   └── profile_*.py
│
├── docs/
│   ├── benchmarking_methodology.md
│   ├── dependency_policy.md
│   ├── gpu_optimization_summary.md
│   ├── performance_regression_policy.md
│   ├── release_process.md
│   ├── repository_conventions.md
│   ├── reproducibility_checklist.md
│   ├── reproducibility_standard.md
│   ├── testing_strategy.md
│   └── git_workflow.md
│
├── environments/
│   ├── cpu_reference_2026-09.txt
│   ├── tesla_t4_reference_2026-09.txt
│   └── README.md
│
├── src/
│   └── quantgpu/
│       ├── backends/
│       ├── benchmarking/
│       ├── pricing/
│       ├── simulation/
│       └── validation/
│
├── tests/
│   ├── integration/
│   ├── numerical/
│   ├── unit/
│   └── conftest.py
│
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Benchmark integrity

QuantGPU separates four concepts that are often conflated in performance work:

```text
execution
    ≠
numerical correctness
    ≠
reproducibility
    ≠
performance
```

Each is validated independently.

Optimizations may change:

- precision
- execution backend
- memory behavior
- operation fusion
- compilation strategy
- kernel implementation

but they may not silently change the financial quantity being computed.

---

## Engineering focus

QuantGPU is designed as both a quantitative finance project and a systems
engineering project.

It demonstrates work across:

- Monte Carlo methods
- option pricing
- numerical validation
- statistical uncertainty
- Python package architecture
- NumPy
- PyTorch
- CUDA
- `torch.compile`
- Triton kernels
- benchmarking methodology
- performance profiling
- reproducible research
- CI and automated testing
- type-safe Python engineering
- release and provenance management

The objective is not simply to show that a GPU is faster than a CPU.

The objective is to build a defensible process for answering:

> **How much faster is this quantitative implementation, and can the result be trusted?**

---

## Documentation

Detailed engineering documentation is available in:

- [Benchmarking methodology](docs/benchmarking_methodology.md)
- [GPU optimization summary](docs/gpu_optimization_summary.md)
- [Testing strategy](docs/testing_strategy.md)
- [Reproducibility standard](docs/reproducibility_standard.md)
- [Reproducibility checklist](docs/reproducibility_checklist.md)
- [Performance regression policy](docs/performance_regression_policy.md)
- [Dependency policy](docs/dependency_policy.md)
- [Repository conventions](docs/repository_conventions.md)
- [Git workflow](docs/git_workflow.md)
- [Release process](docs/release_process.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

---

## License

QuantGPU is released under the MIT License.

See [LICENSE](LICENSE).