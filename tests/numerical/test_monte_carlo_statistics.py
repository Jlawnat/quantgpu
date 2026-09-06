from math import exp, log, sqrt
from statistics import NormalDist

import numpy as np
import pytest

from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.pricing.monte_carlo import price_european_call_mc


def _discounted_call_payoff_std(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    sqrt_t = sqrt(maturity)

    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )

    d2 = d1 - volatility * sqrt_t

    d3 = (log(spot / strike) + (rate + 1.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )

    normal = NormalDist()

    second_moment = (
        spot**2 * exp(volatility**2 * maturity) * normal.cdf(d3)
        - 2.0 * strike * spot * exp(-rate * maturity) * normal.cdf(d1)
        + strike**2 * exp(-2.0 * rate * maturity) * normal.cdf(d2)
    )

    mean = black_scholes_call(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    return sqrt(second_moment - mean**2)


def test_reported_standard_error_matches_theory() -> None:
    n_paths = 500_000

    result = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=n_paths,
        seed=42,
    )

    payoff_std = _discounted_call_payoff_std(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
    )

    theoretical_se = payoff_std / sqrt(n_paths)

    assert result.standard_error == pytest.approx(
        theoretical_se,
        rel=0.02,
    )


def test_reported_standard_error_is_empirically_calibrated() -> None:
    prices = []
    standard_errors = []

    for seed in range(40):
        result = price_european_call_mc(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=25_000,
            seed=seed,
        )

        prices.append(result.price)
        standard_errors.append(result.standard_error)

    empirical_std = float(np.std(prices, ddof=1))
    mean_reported_se = float(np.mean(standard_errors))

    assert empirical_std == pytest.approx(
        mean_reported_se,
        rel=0.15,
    )
