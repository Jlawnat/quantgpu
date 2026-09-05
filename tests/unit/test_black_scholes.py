from math import exp

import pytest

from quantgpu.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


def test_black_scholes_call_reference_value() -> None:
    price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    assert price == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_put_reference_value() -> None:
    price = black_scholes_put(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    assert price == pytest.approx(5.5735, abs=1e-4)


def test_call_put_parity() -> None:
    spot = 100.0
    strike = 110.0
    maturity = 2.0
    rate = 0.03
    volatility = 0.25

    call = black_scholes_call(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    put = black_scholes_put(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    lhs = call - put
    rhs = spot - strike * exp(-rate * maturity)

    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "volatility"),
    [
        (0.0, 100.0, 1.0, 0.2),
        (-1.0, 100.0, 1.0, 0.2),
        (100.0, 0.0, 1.0, 0.2),
        (100.0, -1.0, 1.0, 0.2),
        (100.0, 100.0, -1.0, 0.2),
        (100.0, 100.0, 1.0, -0.2),
    ],
)
def test_invalid_inputs_raise(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    with pytest.raises(ValueError):
        black_scholes_call(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=0.05,
            volatility=volatility,
        )