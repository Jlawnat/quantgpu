from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from quantgpu.backends.numpy_cpu import price_european_call_numpy_cpu
from quantgpu.backends.protocol import PricingResult
from quantgpu.backends.torch_cpu import price_european_call_torch_cpu
from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.benchmarking.cuda_timer import benchmark_cuda_callable
from quantgpu.benchmarking.schema import BENCHMARK_SCHEMA_VERSION
from quantgpu.benchmarking.system_info import get_system_info
from quantgpu.benchmarking.timer import benchmark_callable
from quantgpu.pricing.black_scholes import black_scholes_call

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "cpu_gpu_comparison_v1.csv"

CPU_WARMUP_RUNS = 1
CPU_REPETITIONS = 5

CUDA_WARMUP_RUNS = 3
CUDA_REPETITIONS = 10

PATH_COUNTS = [
    10_000,
    100_000,
    1_000_000,
    5_000_000,
    10_000_000,
]

BackendFunction = Callable[..., PricingResult]


def benchmark_cpu_backend(
    *,
    backend_name: str,
    backend_function: BackendFunction,
    n_paths: int,
    reference_price: float,
) -> dict[str, str | int | float]:
    """Benchmark a CPU pricing backend."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42

    def workload() -> None:
        backend_function(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            n_paths=n_paths,
            seed=seed,
        )

    timing = benchmark_callable(
        workload,
        warmup_runs=CPU_WARMUP_RUNS,
        repetitions=CPU_REPETITIONS,
    )

    result = backend_function(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        n_paths=n_paths,
        seed=seed,
    )

    system_info = get_system_info()

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": backend_name,
        "device": "cpu",
        "python_version": system_info.python_version,
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": n_paths,
        "warmup_runs": CPU_WARMUP_RUNS,
        "repetitions": CPU_REPETITIONS,
        "median_ms": timing.median_seconds * 1_000.0,
        "min_ms": timing.min_seconds * 1_000.0,
        "max_ms": timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": n_paths / timing.median_seconds,
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
        "seed": seed,
    }


def benchmark_cuda_backend(
    *,
    n_paths: int,
    reference_price: float,
) -> dict[str, str | int | float]:
    """Benchmark the PyTorch CUDA pricing backend."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42

    def workload() -> None:
        price_european_call_torch_cuda(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            n_paths=n_paths,
            seed=seed,
        )

    timing = benchmark_cuda_callable(
        workload,
        warmup_runs=CUDA_WARMUP_RUNS,
        repetitions=CUDA_REPETITIONS,
    )

    result = price_european_call_torch_cuda(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        n_paths=n_paths,
        seed=seed,
    )

    system_info = get_system_info()

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": "torch_cuda",
        "device": "cuda",
        "python_version": system_info.python_version,
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": n_paths,
        "warmup_runs": CUDA_WARMUP_RUNS,
        "repetitions": CUDA_REPETITIONS,
        "median_ms": timing.median_seconds * 1_000.0,
        "min_ms": timing.min_seconds * 1_000.0,
        "max_ms": timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": n_paths / timing.median_seconds,
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
        "seed": seed,
    }


def save_results(
    rows: list[dict[str, str | int | float]],
) -> None:
    """Append CPU/GPU benchmark rows to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def main() -> None:
    """Benchmark NumPy CPU, PyTorch CPU, and PyTorch CUDA."""
    reference_price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    rows: list[dict[str, str | int | float]] = []

    print(
        f"{'backend':>12} "
        f"{'paths':>12} "
        f"{'median_ms':>12} "
        f"{'paths/sec':>15} "
        f"{'abs_error':>12}"
    )

    cpu_backends: dict[str, BackendFunction] = {
        "numpy_cpu": price_european_call_numpy_cpu,
        "torch_cpu": price_european_call_torch_cpu,
    }

    for backend_name, backend_function in cpu_backends.items():
        for n_paths in PATH_COUNTS:
            row = benchmark_cpu_backend(
                backend_name=backend_name,
                backend_function=backend_function,
                n_paths=n_paths,
                reference_price=reference_price,
            )

            rows.append(row)

            print(
                f"{backend_name:>12} "
                f"{n_paths:12,d} "
                f"{float(row['median_ms']):12.3f} "
                f"{float(row['throughput_paths_per_sec']):15,.0f} "
                f"{float(row['absolute_error']):12.6f}"
            )

    for n_paths in PATH_COUNTS:
        row = benchmark_cuda_backend(
            n_paths=n_paths,
            reference_price=reference_price,
        )

        rows.append(row)

        print(
            f"{'torch_cuda':>12} "
            f"{n_paths:12,d} "
            f"{float(row['median_ms']):12.3f} "
            f"{float(row['throughput_paths_per_sec']):15,.0f} "
            f"{float(row['absolute_error']):12.6f}"
        )

    save_results(rows)

    print(f"\nSaved benchmark results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()