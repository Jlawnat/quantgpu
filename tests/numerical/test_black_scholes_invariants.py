from math import exp

import pytest

from quantgpu.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


def test_call_respects_no_arbitrage_bounds() -> None:
    spot = 100.0
    strike = 110.0
    maturity = 2.0
    rate = 0.03
    volatility = 0.25

    price = black_scholes_call(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    lower_bound = max(
        spot - strike * exp(-rate * maturity),
        0.0,
    )

    assert lower_bound <= price <= spot


def test_put_respects_no_arbitrage_bounds() -> None:
    spot = 100.0
    strike = 110.0
    maturity = 2.0
    rate = 0.03
    volatility = 0.25

    price = black_scholes_put(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    discounted_strike = strike * exp(-rate * maturity)
    lower_bound = max(discounted_strike - spot, 0.0)

    assert lower_bound <= price <= discounted_strike


def test_call_price_increases_with_spot() -> None:
    low = black_scholes_call(90.0, 100.0, 1.0, 0.05, 0.20)
    high = black_scholes_call(110.0, 100.0, 1.0, 0.05, 0.20)

    assert high > low


def test_call_price_decreases_with_strike() -> None:
    low_strike = black_scholes_call(
        100.0,
        90.0,
        1.0,
        0.05,
        0.20,
    )

    high_strike = black_scholes_call(
        100.0,
        110.0,
        1.0,
        0.05,
        0.20,
    )

    assert low_strike > high_strike


def test_call_price_increases_with_volatility() -> None:
    low_volatility = black_scholes_call(
        100.0,
        100.0,
        1.0,
        0.05,
        0.10,
    )

    high_volatility = black_scholes_call(
        100.0,
        100.0,
        1.0,
        0.05,
        0.40,
    )

    assert high_volatility > low_volatility


def test_black_scholes_is_homogeneous_in_spot_and_strike() -> None:
    scale = 3.0

    base = black_scholes_call(
        100.0,
        110.0,
        1.0,
        0.05,
        0.20,
    )

    scaled = black_scholes_call(
        scale * 100.0,
        scale * 110.0,
        1.0,
        0.05,
        0.20,
    )

    assert scaled == pytest.approx(
        scale * base,
        rel=1e-12,
    )
def test_call_at_expiry_equals_intrinsic_value() -> None:
    price = black_scholes_call(
        120.0,
        100.0,
        0.0,
        0.05,
        0.20,
    )

    assert price == 20.0


def test_put_at_expiry_equals_intrinsic_value() -> None:
    price = black_scholes_put(
        80.0,
        100.0,
        0.0,
        0.05,
        0.20,
    )

    assert price == 20.0


def test_call_zero_volatility_matches_deterministic_value() -> None:
    spot = 100.0
    strike = 90.0
    maturity = 1.0
    rate = 0.05

    price = black_scholes_call(
        spot,
        strike,
        maturity,
        rate,
        0.0,
    )

    expected = max(
        spot - strike * exp(-rate * maturity),
        0.0,
    )

    assert price == pytest.approx(expected, abs=1e-12)


def test_put_zero_volatility_matches_deterministic_value() -> None:
    spot = 80.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05

    price = black_scholes_put(
        spot,
        strike,
        maturity,
        rate,
        0.0,
    )

    expected = max(
        strike * exp(-rate * maturity) - spot,
        0.0,
    )

    assert price == pytest.approx(expected, abs=1e-12)