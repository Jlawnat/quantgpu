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


def simulate_gbm_terminal_torch_cuda(
    *,
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    n_paths: int,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Simulate terminal GBM prices using PyTorch on CUDA."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    seed = validate_seed(seed)
    device = _require_cuda()

    if maturity == 0:
        return torch.full(
            (n_paths,),
            spot,
            dtype=dtype,
            device=device,
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

    exponent = (
        (drift - 0.5 * volatility**2) * maturity
        + volatility * sqrt(maturity) * z
    )

    return spot * torch.exp(exponent)


def price_european_call_torch_cuda(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> PricingResult:
    """Price a European call using PyTorch CUDA Monte Carlo."""
    if strike <= 0:
        raise ValueError("strike must be positive")

    terminal_prices = simulate_gbm_terminal_torch_cuda(
        spot=spot,
        drift=rate,
        volatility=volatility,
        maturity=maturity,
        n_paths=n_paths,
        seed=seed,
        dtype=dtype,
    )

    payoffs = torch.clamp(
        terminal_prices - strike,
        min=0.0,
    )

    discounted_payoffs = exp(-rate * maturity) * payoffs

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