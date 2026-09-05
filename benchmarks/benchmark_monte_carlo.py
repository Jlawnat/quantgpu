from __future__ import annotations

from quantgpu.benchmarking.timer import benchmark_callable
from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.pricing.monte_carlo import price_european_call_mc


def main() -> None:
    """Benchmark NumPy Monte Carlo pricing across several path counts."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    seed = 42

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


if __name__ == "__main__":
    main()