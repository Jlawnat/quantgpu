from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from quantgpu.backends.numpy_cpu import price_european_call_numpy_cpu
from quantgpu.backends.protocol import PricingResult
from quantgpu.backends.torch_cpu import price_european_call_torch_cpu
from quantgpu.benchmarking.environment import get_software_environment
from quantgpu.benchmarking.provenance import get_source_provenance
from quantgpu.benchmarking.schema import (
    BENCHMARK_SCHEMA_VERSION,
    validate_benchmark_metadata,
)
from quantgpu.benchmarking.system_info import get_system_info
from quantgpu.benchmarking.timer import benchmark_callable
from quantgpu.pricing.black_scholes import black_scholes_call

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "backend_comparison_v2.csv"

WARMUP_RUNS = 1
REPETITIONS = 5

PATH_COUNTS = [
    10_000,
    100_000,
    1_000_000,
    5_000_000,
    10_000_000,
]

BackendFunction = Callable[..., PricingResult]

BACKENDS: dict[str, BackendFunction] = {
    "numpy_cpu": price_european_call_numpy_cpu,
    "torch_cpu": price_european_call_torch_cpu,
}


def benchmark_backend(
    *,
    backend_name: str,
    backend_function: BackendFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int,
    reference_price: float,
) -> dict[str, str | int | float]:
    """Benchmark one pricing backend for one workload size."""

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
        warmup_runs=WARMUP_RUNS,
        repetitions=REPETITIONS,
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

    median_ms = timing.median_seconds * 1_000.0
    throughput = n_paths / timing.median_seconds
    absolute_error = abs(result.price - reference_price)

    system_info = get_system_info()
    provenance = get_source_provenance()
    software = get_software_environment()

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": provenance.git_commit,
        "git_tree_state": provenance.git_tree_state,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": backend_name,
        "device": "cpu",
        "dtype": "float64",
        "python_version": software.python_version,
        "quantgpu_version": software.quantgpu_version,
        "numpy_version": software.numpy_version,
        "torch_version": software.torch_version,
        "cuda_version": software.cuda_version,
        "triton_version": software.triton_version,
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": n_paths,
        "warmup_runs": WARMUP_RUNS,
        "repetitions": REPETITIONS,
        "median_ms": median_ms,
        "min_ms": timing.min_seconds * 1_000.0,
        "max_ms": timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": throughput,
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": absolute_error,
        "standard_error": result.standard_error,
        "seed": seed,
    }


def save_results(
    rows: list[dict[str, str | int | float]],
) -> None:
    """Append backend comparison results to CSV."""
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
    """Benchmark all registered CPU pricing backends."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42

    reference_price = black_scholes_call(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    rows: list[dict[str, str | int | float]] = []

    print(
        f"{'backend':>12} "
        f"{'paths':>12} "
        f"{'median_ms':>12} "
        f"{'paths/sec':>15} "
        f"{'abs_error':>12}"
    )

    for backend_name, backend_function in BACKENDS.items():
        for n_paths in PATH_COUNTS:
            row = benchmark_backend(
                backend_name=backend_name,
                backend_function=backend_function,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                volatility=volatility,
                n_paths=n_paths,
                seed=seed,
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

    save_results(rows)

    print(f"\nSaved benchmark results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()