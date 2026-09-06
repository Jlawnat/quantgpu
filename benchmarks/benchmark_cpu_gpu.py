from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from quantgpu.backends.numpy_cpu import price_european_call_numpy_cpu
from quantgpu.backends.protocol import PricingResult
from quantgpu.backends.torch_cpu import price_european_call_torch_cpu
from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.benchmarking.cuda_timer import benchmark_cuda_callable
from quantgpu.benchmarking.environment import get_software_environment
from quantgpu.benchmarking.provenance import get_source_provenance
from quantgpu.benchmarking.schema import (
    BENCHMARK_SCHEMA_VERSION,
    validate_benchmark_metadata,
)
from quantgpu.benchmarking.system_info import get_system_info
from quantgpu.benchmarking.timer import benchmark_callable
from quantgpu.benchmarking.validation import require_valid_result
from quantgpu.pricing.black_scholes import black_scholes_call

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "cpu_gpu_comparison_v3.csv"

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


def _common_parameters() -> dict[str, float | int]:
    """Return the canonical benchmark workload parameters."""
    return {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "seed": 42,
    }


def benchmark_cpu_backend(
    *,
    backend_name: str,
    backend_function: BackendFunction,
    n_paths: int,
    reference_price: float,
) -> dict[str, str | int | float]:
    """Benchmark one CPU pricing backend."""
    params = _common_parameters()

    def workload() -> None:
        backend_function(
            spot=float(params["spot"]),
            strike=float(params["strike"]),
            maturity=float(params["maturity"]),
            rate=float(params["rate"]),
            volatility=float(params["volatility"]),
            n_paths=n_paths,
            seed=int(params["seed"]),
        )

    timing = benchmark_callable(
        workload,
        warmup_runs=CPU_WARMUP_RUNS,
        repetitions=CPU_REPETITIONS,
    )

    result = backend_function(
        spot=float(params["spot"]),
        strike=float(params["strike"]),
        maturity=float(params["maturity"]),
        rate=float(params["rate"]),
        volatility=float(params["volatility"]),
        n_paths=n_paths,
        seed=int(params["seed"]),
    )

    require_valid_result(result, reference_price)

    system_info = get_system_info()
    provenance = get_source_provenance()
    software = get_software_environment()

    median_ms = timing.median_seconds * 1_000.0

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": provenance.git_commit,
        "git_tree_state": provenance.git_tree_state,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": backend_name,
        "device": "cpu",
        "dtype": "float64",
        "spot": float(params["spot"]),
        "strike": float(params["strike"]),
        "maturity": float(params["maturity"]),
        "rate": float(params["rate"]),
        "volatility": float(params["volatility"]),
        "python_version": software.python_version,
        "quantgpu_version": software.quantgpu_version,
        "numpy_version": software.numpy_version,
        "torch_version": software.torch_version,
        "cuda_version": software.cuda_version,
        "triton_version": software.triton_version,
        "gpu_name": "none",
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": n_paths,
        "warmup_runs": CPU_WARMUP_RUNS,
        "repetitions": CPU_REPETITIONS,
        "device_median_ms": median_ms,
        "end_to_end_median_ms": median_ms,
        "min_ms": timing.min_seconds * 1_000.0,
        "max_ms": timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": n_paths / timing.median_seconds,
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
        "validation_status": "passed",
        "seed": int(params["seed"]),
    }


def benchmark_cuda_backend(
    *,
    n_paths: int,
    reference_price: float,
) -> dict[str, str | int | float]:
    """Benchmark PyTorch CUDA using device and end-to-end timing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    params = _common_parameters()

    def workload() -> None:
        price_european_call_torch_cuda(
            spot=float(params["spot"]),
            strike=float(params["strike"]),
            maturity=float(params["maturity"]),
            rate=float(params["rate"]),
            volatility=float(params["volatility"]),
            n_paths=n_paths,
            seed=int(params["seed"]),
        )

    device_timing = benchmark_cuda_callable(
        workload,
        warmup_runs=CUDA_WARMUP_RUNS,
        repetitions=CUDA_REPETITIONS,
    )

    for _ in range(CUDA_WARMUP_RUNS):
        workload()

    torch.cuda.synchronize()

    wall_times: list[float] = []

    for _ in range(CUDA_REPETITIONS):
        start = perf_counter()

        workload()
        torch.cuda.synchronize()

        wall_times.append(perf_counter() - start)

    wall_median_seconds = median(wall_times)

    result = price_european_call_torch_cuda(
        spot=float(params["spot"]),
        strike=float(params["strike"]),
        maturity=float(params["maturity"]),
        rate=float(params["rate"]),
        volatility=float(params["volatility"]),
        n_paths=n_paths,
        seed=int(params["seed"]),
    )

    require_valid_result(result, reference_price)

    system_info = get_system_info()
    provenance = get_source_provenance()
    software = get_software_environment()
    gpu_name = torch.cuda.get_device_name(0)

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": provenance.git_commit,
        "git_tree_state": provenance.git_tree_state,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": "torch_cuda",
        "device": "cuda",
        "dtype": "float64",
        "python_version": software.python_version,
        "quantgpu_version": software.quantgpu_version,
        "numpy_version": software.numpy_version,
        "torch_version": software.torch_version,
        "cuda_version": software.cuda_version,
        "triton_version": software.triton_version,
        "gpu_name": gpu_name,
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": n_paths,
        "warmup_runs": CUDA_WARMUP_RUNS,
        "repetitions": CUDA_REPETITIONS,
        "device_median_ms": device_timing.median_seconds * 1_000.0,
        "end_to_end_median_ms": wall_median_seconds * 1_000.0,
        "min_ms": device_timing.min_seconds * 1_000.0,
        "max_ms": device_timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": (
            n_paths / device_timing.median_seconds
        ),
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
        "validation_status": "passed",
        "seed": int(params["seed"]),
    }


def add_speedups(
    rows: list[dict[str, str | int | float]],
) -> None:
    """Add speedups versus NumPy and PyTorch CPU."""
    for n_paths in PATH_COUNTS:
        matching = [
            row
            for row in rows
            if int(row["n_paths"]) == n_paths
        ]

        numpy_row = next(
            row for row in matching if row["backend"] == "numpy_cpu"
        )

        torch_cpu_row = next(
            row for row in matching if row["backend"] == "torch_cpu"
        )

        numpy_ms = float(numpy_row["end_to_end_median_ms"])
        torch_cpu_ms = float(torch_cpu_row["end_to_end_median_ms"])

        for row in matching:
            row_ms = float(row["end_to_end_median_ms"])

            row["speedup_vs_numpy"] = numpy_ms / row_ms
            row["speedup_vs_torch_cpu"] = torch_cpu_ms / row_ms


def save_results(
    rows: list[dict[str, str | int | float]],
) -> None:
    """Append benchmark rows to the versioned CSV."""
    if not rows:
        raise ValueError("rows must not be empty")
    for row in rows:
        validate_benchmark_metadata(row)

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
    params = _common_parameters()

    reference_price = black_scholes_call(
        spot=float(params["spot"]),
        strike=float(params["strike"]),
        maturity=float(params["maturity"]),
        rate=float(params["rate"]),
        volatility=float(params["volatility"]),
    )

    rows: list[dict[str, str | int | float]] = []

    cpu_backends: dict[str, BackendFunction] = {
        "numpy_cpu": price_european_call_numpy_cpu,
        "torch_cpu": price_european_call_torch_cpu,
    }

    for backend_name, backend_function in cpu_backends.items():
        for n_paths in PATH_COUNTS:
            rows.append(
                benchmark_cpu_backend(
                    backend_name=backend_name,
                    backend_function=backend_function,
                    n_paths=n_paths,
                    reference_price=reference_price,
                )
            )

    for n_paths in PATH_COUNTS:
        rows.append(
            benchmark_cuda_backend(
                n_paths=n_paths,
                reference_price=reference_price,
            )
        )

    add_speedups(rows)

    print(
        f"{'backend':>12} "
        f"{'paths':>12} "
        f"{'device_ms':>12} "
        f"{'e2e_ms':>12} "
        f"{'vs_numpy':>10} "
        f"{'vs_torch':>10}"
    )

    for row in rows:
        print(
            f"{str(row['backend']):>12} "
            f"{int(row['n_paths']):12,d} "
            f"{float(row['device_median_ms']):12.3f} "
            f"{float(row['end_to_end_median_ms']):12.3f} "
            f"{float(row['speedup_vs_numpy']):10.2f} "
            f"{float(row['speedup_vs_torch_cpu']):10.2f}"
        )

    save_results(rows)

    print(f"\nSaved benchmark results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
