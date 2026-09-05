from __future__ import annotations

from statistics import median
from time import perf_counter

import torch

from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.pricing.black_scholes import black_scholes_call

PATH_COUNTS = [
    100_000,
    1_000_000,
    5_000_000,
    10_000_000,
]

WARMUP_RUNS = 3
REPETITIONS = 10


def benchmark_dtype(
    *,
    dtype: torch.dtype,
    n_paths: int,
    reference_price: float,
) -> dict[str, float | int | str]:
    """Benchmark one CUDA dtype at one path count."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    def workload() -> None:
        price_european_call_torch_cuda(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=n_paths,
            seed=42,
            dtype=dtype,
        )

    for _ in range(WARMUP_RUNS):
        workload()

    torch.cuda.synchronize()

    timings: list[float] = []

    for _ in range(REPETITIONS):
        start = perf_counter()

        workload()
        torch.cuda.synchronize()

        timings.append(perf_counter() - start)

    result = price_european_call_torch_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=n_paths,
        seed=42,
        dtype=dtype,
    )

    median_seconds = median(timings)

    return {
        "dtype": str(dtype),
        "n_paths": n_paths,
        "median_ms": median_seconds * 1_000.0,
        "throughput": n_paths / median_seconds,
        "price": result.price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
    }


def main() -> None:
    """Compare float32 and float64 CUDA Monte Carlo performance."""
    reference_price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    print(
        f"{'dtype':>14} "
        f"{'paths':>12} "
        f"{'median_ms':>12} "
        f"{'paths/sec':>15} "
        f"{'abs_error':>12} "
        f"{'std_error':>12}"
    )

    for dtype in (torch.float64, torch.float32):
        for n_paths in PATH_COUNTS:
            row = benchmark_dtype(
                dtype=dtype,
                n_paths=n_paths,
                reference_price=reference_price,
            )

            print(
                f"{str(row['dtype']):>14} "
                f"{int(row['n_paths']):12,d} "
                f"{float(row['median_ms']):12.3f} "
                f"{float(row['throughput']):15,.0f} "
                f"{float(row['absolute_error']):12.6f} "
                f"{float(row['standard_error']):12.6f}"
            )


if __name__ == "__main__":
    main()