from __future__ import annotations

import csv
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantgpu.benchmarking.timer import benchmark_callable
from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.pricing.monte_carlo import price_european_call_mc
from quantgpu.benchmarking.system_info import get_system_info

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "monte_carlo_numpy.csv"


def main() -> None:
    """Benchmark NumPy Monte Carlo pricing across several path counts."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42
    system_info = get_system_info()

    reference = black_scholes_call(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    path_counts = [
        10_000,
        100_000,
        1_000_000,
    ]

    rows: list[dict[str, str | int | float]] = []

    print(
        f"{'paths':>12} "
        f"{'median_ms':>12} "
        f"{'paths/sec':>15} "
        f"{'abs_error':>12}"
    )

    for n_paths in path_counts:

        def workload() -> None:
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
            warmup_runs=1,
            repetitions=5,
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

        median_ms = timing.median_seconds * 1_000.0
        throughput = n_paths / timing.median_seconds
        absolute_error = abs(result.price - reference)

        print(
            f"{n_paths:12,d} "
            f"{median_ms:12.3f} "
            f"{throughput:15,.0f} "
            f"{absolute_error:12.6f}"
        )

        rows.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "backend": "numpy",
                "device": "cpu",
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "n_paths": n_paths,
                "warmup_runs": 1,
                "repetitions": 5,
                "median_ms": median_ms,
                "min_ms": timing.min_seconds * 1_000.0,
                "max_ms": timing.max_seconds * 1_000.0,
                "throughput_paths_per_sec": throughput,
                "estimated_price": result.price,
                "reference_price": reference,
                "absolute_error": absolute_error,
                "standard_error": result.standard_error,
                "seed": seed,
                "os": system_info.os,
                "os_release": system_info.os_release,
                "machine": system_info.machine,
                "processor": system_info.processor,
                "cpu_model": system_info.cpu_model,
            }
        )

    save_results(rows)


def save_results(rows: list[dict[str, str | int | float]]) -> None:
    """Append benchmark rows to the CSV results file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"\nSaved benchmark results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()