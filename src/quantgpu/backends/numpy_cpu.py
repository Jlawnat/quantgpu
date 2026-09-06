from __future__ import annotations

from quantgpu.backends.protocol import PricingResult
from quantgpu.pricing.monte_carlo import price_european_call_mc


def price_european_call_numpy_cpu(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = None,
) -> PricingResult:
    """Price a European call using the NumPy CPU backend."""
    result = price_european_call_mc(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        n_paths=n_paths,
        seed=seed,
    )

    return PricingResult(
        price=result.price,
        standard_error=result.standard_error,
        n_paths=result.n_paths,
    )
