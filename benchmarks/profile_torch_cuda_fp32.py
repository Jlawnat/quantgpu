from __future__ import annotations

from math import exp, sqrt
from statistics import median

import torch

WARMUP_RUNS = 3
REPETITIONS = 10

SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20
N_PATHS = 10_000_000
SEED = 42

DTYPE = torch.float32
DEVICE = torch.device("cuda")


def time_cuda_stage(function) -> float:
    """Return median CUDA device time in milliseconds."""
    for _ in range(WARMUP_RUNS):
        function()

    torch.cuda.synchronize()

    timings_ms: list[float] = []

    for _ in range(REPETITIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        function()
        end.record()

        torch.cuda.synchronize()

        timings_ms.append(start.elapsed_time(end))

    return median(timings_ms)


def main() -> None:
    """Profile the FP32 PyTorch CUDA Monte Carlo pipeline."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    sqrt_t = sqrt(MATURITY)
    drift_term = (RATE - 0.5 * VOLATILITY**2) * MATURITY
    diffusion_scale = VOLATILITY * sqrt_t
    discount_factor = exp(-RATE * MATURITY)

    z = torch.empty(
        N_PATHS,
        dtype=DTYPE,
        device=DEVICE,
    )

    terminal_prices = torch.empty_like(z)
    payoffs = torch.empty_like(z)
    discounted_payoffs = torch.empty_like(z)

    def rng_stage() -> None:
        z.normal_(generator=generator)

    def gbm_stage() -> None:
        torch.mul(
            z,
            diffusion_scale,
            out=terminal_prices,
        )
        terminal_prices.add_(drift_term)
        terminal_prices.exp_()
        terminal_prices.mul_(SPOT)

    def payoff_stage() -> None:
        torch.sub(
            terminal_prices,
            STRIKE,
            out=payoffs,
        )
        payoffs.clamp_(min=0.0)

    def discount_stage() -> None:
        torch.mul(
            payoffs,
            discount_factor,
            out=discounted_payoffs,
        )

    def mean_stage() -> None:
        discounted_payoffs.mean()

    def std_stage() -> None:
        discounted_payoffs.std(unbiased=True)

    rng_ms = time_cuda_stage(rng_stage)

    rng_stage()
    torch.cuda.synchronize()

    gbm_ms = time_cuda_stage(gbm_stage)

    gbm_stage()
    torch.cuda.synchronize()

    payoff_ms = time_cuda_stage(payoff_stage)

    payoff_stage()
    torch.cuda.synchronize()

    discount_ms = time_cuda_stage(discount_stage)

    discount_stage()
    torch.cuda.synchronize()

    mean_ms = time_cuda_stage(mean_stage)
    std_ms = time_cuda_stage(std_stage)

    total_component_ms = (
        rng_ms
        + gbm_ms
        + payoff_ms
        + discount_ms
        + mean_ms
        + std_ms
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Paths: {N_PATHS:,}")
    print(f"Dtype: {DTYPE}")
    print()
    print(f"{'stage':<20} {'median_ms':>12} {'share_%':>10}")

    stages = [
        ("rng", rng_ms),
        ("gbm_transform", gbm_ms),
        ("payoff", payoff_ms),
        ("discount", discount_ms),
        ("mean_reduction", mean_ms),
        ("std_reduction", std_ms),
    ]

    for name, timing_ms in stages:
        share = (
            timing_ms / total_component_ms * 100.0
            if total_component_ms > 0
            else 0.0
        )

        print(
            f"{name:<20} "
            f"{timing_ms:12.3f} "
            f"{share:10.2f}"
        )

    print()
    print(
        f"{'component_sum':<20} "
        f"{total_component_ms:12.3f}"
    )


if __name__ == "__main__":
    main()