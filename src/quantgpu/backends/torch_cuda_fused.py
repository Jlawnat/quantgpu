from __future__ import annotations

from math import exp, sqrt

import torch

from quantgpu.backends.protocol import PricingResult
from quantgpu.simulation.rng import validate_seed

def _require_cuda() -> torch.device:
    """Return the CUDA device or raise if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")

    return torch.device("cuda")


def price_european_call_torch_cuda_fused(
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
    """Price a European call using a reduced-intermediate CUDA pipeline."""
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
    seed = validate_seed(seed)
    device = _require_cuda()

    if maturity == 0:
        payoff = max(spot - strike, 0.0)
        return PricingResult(
            price=payoff,
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
        dtype=dtype,
        device=device,
    )

    sqrt_t = sqrt(maturity)
    drift_term = (rate - 0.5 * volatility**2) * maturity
    diffusion_scale = volatility * sqrt_t
    discount_factor = exp(-rate * maturity)

    discounted_payoffs = (
        torch.exp(z * diffusion_scale + drift_term)
        .mul(spot)
        .sub(strike)
        .clamp_min(0.0)
        .mul(discount_factor)
    )

    price = float(discounted_payoffs.mean().item())

    if n_paths == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            (
                discounted_payoffs.std(unbiased=True)
                / sqrt(n_paths)
            ).item()
        )

    return PricingResult(
        price=price,
        standard_error=standard_error,
        n_paths=n_paths,
    )