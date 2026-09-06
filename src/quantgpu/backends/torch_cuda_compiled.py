from __future__ import annotations

from math import exp, sqrt

import torch

from quantgpu.backends.protocol import PricingResult
from quantgpu.simulation.rng import validate_seed


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")

    return torch.device("cuda")


@torch.compile(mode="reduce-overhead")
def _compiled_payoff_stats(
    z: torch.Tensor,
    spot: float,
    strike: float,
    drift_term: float,
    diffusion_scale: float,
    discount_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    discounted_payoffs = (
        torch.exp(z * diffusion_scale + drift_term)
        .mul(spot)
        .sub(strike)
        .clamp_min(0.0)
        .mul(discount_factor)
    )

    return (
        discounted_payoffs.mean(),
        discounted_payoffs.std(unbiased=True),
    )


def price_european_call_torch_cuda_compiled(
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
        dtype=dtype,
        device=device,
    )

    drift_term = (rate - 0.5 * volatility**2) * maturity
    diffusion_scale = volatility * sqrt(maturity)
    discount_factor = exp(-rate * maturity)

    mean, std = _compiled_payoff_stats(
        z,
        spot,
        strike,
        drift_term,
        diffusion_scale,
        discount_factor,
    )

    price = float(mean.item())

    if n_paths == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            (std / sqrt(n_paths)).item()
        )

    return PricingResult(
        price=price,
        standard_error=standard_error,
        n_paths=n_paths,
    )