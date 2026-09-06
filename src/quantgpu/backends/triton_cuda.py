from __future__ import annotations

from math import exp, sqrt

import torch
import triton
import triton.language as tl

from quantgpu.backends.protocol import PricingResult
from quantgpu.simulation.rng import validate_seed

BLOCK_SIZE = 1024


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")

    return torch.device("cuda")


@triton.jit
def _payoff_moments_kernel(
    z_ptr,
    partial_sum_ptr,
    partial_square_ptr,
    n_paths,
    spot,
    strike,
    drift_term,
    diffusion_scale,
    discount_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_paths

    z = tl.load(
        z_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    terminal = spot * tl.exp(
        drift_term + diffusion_scale * z
    )

    payoff = tl.maximum(
        terminal - strike,
        0.0,
    ) * discount_factor

    payoff = tl.where(
        mask,
        payoff,
        0.0,
    )

    block_sum = tl.sum(
        payoff,
        axis=0,
    )

    block_square_sum = tl.sum(
        payoff * payoff,
        axis=0,
    )

    tl.store(
        partial_sum_ptr + pid,
        block_sum,
    )

    tl.store(
        partial_square_ptr + pid,
        block_square_sum,
    )


def price_european_call_triton_cuda(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> PricingResult:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if dtype is not torch.float32:
        raise ValueError("Triton backend currently supports float32 only")

    seed = validate_seed(seed)
    device = _require_cuda()

    if maturity == 0:
        return PricingResult(
            price=max(spot - strike, 0.0),
            standard_error=0.0,
            n_paths=n_paths,
        )

    generator = torch.Generator(device=device)

    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    z = torch.randn(
        n_paths,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    n_blocks = triton.cdiv(
        n_paths,
        BLOCK_SIZE,
    )

    partial_sums = torch.empty(
        n_blocks,
        dtype=torch.float32,
        device=device,
    )

    partial_square_sums = torch.empty_like(
        partial_sums,
    )

    drift_term = (
        rate - 0.5 * volatility**2
    ) * maturity

    diffusion_scale = volatility * sqrt(maturity)
    discount_factor = exp(-rate * maturity)

    _payoff_moments_kernel[(n_blocks,)](
        z,
        partial_sums,
        partial_square_sums,
        n_paths,
        spot,
        strike,
        drift_term,
        diffusion_scale,
        discount_factor,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    total = partial_sums.sum(
        dtype=torch.float64,
    )

    total_square = partial_square_sums.sum(
        dtype=torch.float64,
    )

    price = float(
        (total / n_paths).item()
    )

    if n_paths == 1:
        standard_error = 0.0
    else:
        variance_numerator = (
            total_square
            - total * total / n_paths
        )

        sample_variance = torch.clamp(
            variance_numerator / (n_paths - 1),
            min=0.0,
        )

        standard_error = float(
            torch.sqrt(
                sample_variance / n_paths
            ).item()
        )

    return PricingResult(
        price=price,
        standard_error=standard_error,
        n_paths=n_paths,
    )