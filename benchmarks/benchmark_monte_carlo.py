from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

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
from quantgpu.pricing.monte_carlo import price_european_call_mc

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "monte_carlo_numpy_v2.csv"

BACKEND = "numpy"
DEVICE = "cpu"

WARMUP_RUNS = 1
REPETITIONS = 5

PATH_COUNTS = [
    10_000,
    100_000,
    1_000_000,
]


def run_benchmark() -> list[dict[str, str | int | float]]:
    """Run the NumPy Monte Carlo benchmark suite."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42

    system_info = get_system_info()
    provenance = get_source_provenance()
    software = get_software_environment()

    reference_price = black_scholes_call(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    rows: list[dict[str, str | int | float]] = []

    print(
        f"{'paths':>12} "
        f"{'median_ms':>12} "
        f"{'paths/sec':>15} "
        f"{'abs_error':>12}"
    )

    for n_paths in PATH_COUNTS:

        def workload(n_paths: int = n_paths) -> None:
            price_european_call_mc(
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

        result = price_european_call_mc(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            n_paths=n_paths,
            seed=seed,
        )
        require_valid_result(result, reference_price)

        median_ms = timing.median_seconds * 1_000.0
        min_ms = timing.min_seconds * 1_000.0
        max_ms = timing.max_seconds * 1_000.0

        throughput = n_paths / timing.median_seconds
        absolute_error = abs(result.price - reference_price)

        print(
            f"{n_paths:12,d} "
            f"{median_ms:12.3f} "
            f"{throughput:15,.0f} "
            f"{absolute_error:12.6f}"
        )

        rows.append(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "git_commit": provenance.git_commit,
                "git_tree_state": provenance.git_tree_state,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "backend": BACKEND,
                "device": DEVICE,
                "dtype": "float64",
                "spot": spot,
                "strike": strike,
                "maturity": maturity,
                "rate": rate,
                "volatility": volatility,
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
                "min_ms": min_ms,
                "max_ms": max_ms,
                "throughput_paths_per_sec": throughput,
                "estimated_price": result.price,
                "reference_price": reference_price,
                "absolute_error": absolute_error,
                "standard_error": result.standard_error,
                "validation_status": "passed",
                "seed": seed,
            }
        )

    return rows


def save_results(
    rows: list[dict[str, str | int | float]],
) -> None:
    """Append benchmark results to the versioned CSV output."""
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

    print(f"\nSaved benchmark results to {RESULTS_FILE}")


def main() -> None:
    """Run the benchmark suite and persist its results."""
    rows = run_benchmark()
    save_results(rows)


if __name__ == "__main__":
    main()
