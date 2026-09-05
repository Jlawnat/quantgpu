from __future__ import annotations

from statistics import median
from time import perf_counter

import torch

from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.backends.torch_cuda_fused import (
    price_european_call_torch_cuda_fused,
)

N_PATHS = 10_000_000
WARMUP_RUNS = 3
REPETITIONS = 10


def benchmark(function) -> float:
    """Return median synchronized wall time in milliseconds."""
    for _ in range(WARMUP_RUNS):
        function()

    torch.cuda.synchronize()

    timings: list[float] = []

    for _ in range(REPETITIONS):
        start = perf_counter()

        function()
        torch.cuda.synchronize()

        timings.append(perf_counter() - start)

    return median(timings) * 1_000.0


def main() -> None:
    """Compare baseline and fused FP32 CUDA implementations."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    def baseline() -> None:
        price_european_call_torch_cuda(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=42,
            dtype=torch.float32,
        )

    def fused() -> None:
        price_european_call_torch_cuda_fused(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=42,
            dtype=torch.float32,
        )

    baseline_ms = benchmark(baseline)
    fused_ms = benchmark(fused)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Paths: {N_PATHS:,}")
    print(f"Baseline FP32: {baseline_ms:.3f} ms")
    print(f"Fused FP32:    {fused_ms:.3f} ms")
    print(f"Speedup:       {baseline_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()