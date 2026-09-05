from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from quantgpu.simulation.gbm import simulate_gbm_terminal


@dataclass(frozen=True)
class MonteCarloResult:
    """Summary of a Monte Carlo pricing estimate."""

    price: float
    standard_error: float
    n_paths: int


def price_european_call_mc(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = None,
) -> MonteCarloResult:
    """Price a European call option using Monte Carlo simulation."""
    if strike <= 0:
        raise ValueError("strike must be positive")

    terminal_prices = simulate_gbm_terminal(
        spot=spot,
        drift=rate,
        volatility=volatility,
        maturity=maturity,
        n_paths=n_paths,
        seed=seed,
    )

    payoffs = np.maximum(terminal_prices - strike, 0.0)
    discount_factor = exp(-rate * maturity)

    discounted_payoffs = discount_factor * payoffs

    price = float(np.mean(discounted_payoffs))

    if n_paths == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            np.std(discounted_payoffs, ddof=1) / sqrt(n_paths)
        )

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        n_paths=n_paths,
    )