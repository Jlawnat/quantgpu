from __future__ import annotations

from statistics import median
from time import perf_counter

import torch

from quantgpu.backends.torch_cuda import price_european_call_torch_cuda

N_PATHS = 10_000_000
WARMUP_RUNS = 3
REPETITIONS = 10


def compiled_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FP32 Monte Carlo payoff pipeline on CUDA."""
    device = torch.device("cuda")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    z = torch.randn(
        n_paths,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    drift_term = (rate - 0.5 * volatility**2) * maturity
    diffusion_scale = volatility * maturity**0.5
    discount_factor = torch.exp(
        torch.tensor(
            -rate * maturity,
            dtype=torch.float32,
            device=device,
        )
    )

    discounted_payoffs = (
        torch.exp(z * diffusion_scale + drift_term)
        .mul(spot)
        .sub(strike)
        .clamp_min(0.0)
        .mul(discount_factor)
    )

    return discounted_payoffs.mean(), discounted_payoffs.std(unbiased=True)


compiled_price_fn = torch.compile(
    compiled_price,
    mode="reduce-overhead",
)


def benchmark(function) -> float:
    """Return synchronized median wall time in milliseconds."""
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
    """Compare eager and compiled FP32 CUDA execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    def eager() -> None:
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

    def compiled() -> None:
        compiled_price_fn(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=42,
        )

    eager_ms = benchmark(eager)
    compiled_ms = benchmark(compiled)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Paths: {N_PATHS:,}")
    print(f"Eager FP32:    {eager_ms:.3f} ms")
    print(f"Compiled FP32: {compiled_ms:.3f} ms")
    print(f"Speedup:       {eager_ms / compiled_ms:.2f}x")


if __name__ == "__main__":
    main()
