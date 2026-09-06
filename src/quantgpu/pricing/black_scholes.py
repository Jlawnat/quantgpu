from __future__ import annotations

from math import erf, exp, log, sqrt


def _normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Price a European call option using the Black-Scholes formula."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    if maturity == 0:
        return max(spot - strike, 0.0)

    if volatility == 0:
        discounted_strike = strike * exp(-rate * maturity)
        return max(spot - discounted_strike, 0.0)

    sqrt_t = sqrt(maturity)

    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )

    d2 = d1 - volatility * sqrt_t

    return spot * _normal_cdf(d1) - strike * exp(-rate * maturity) * _normal_cdf(d2)


def black_scholes_put(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Price a European put option using the Black-Scholes formula."""
    call_price = black_scholes_call(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    return call_price - spot + strike * exp(-rate * maturity)
