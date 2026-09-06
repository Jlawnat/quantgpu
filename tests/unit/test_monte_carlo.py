import pytest

from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.pricing.monte_carlo import price_european_call_mc
from quantgpu.validation.tolerances import monte_carlo_tolerance


def test_monte_carlo_price_matches_black_scholes() -> None:
    result = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    reference = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    assert result.price == pytest.approx(
        reference,
        rel=monte_carlo_tolerance(result.standard_error),   
    )


def test_monte_carlo_result_reports_standard_error() -> None:
    result = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
    )

    assert result.standard_error > 0.0
    assert result.n_paths == 100_000


def test_monte_carlo_is_reproducible_with_same_seed() -> None:
    first = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=50_000,
        seed=42,
    )

    second = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=50_000,
        seed=42,
    )

    assert first == second
def test_monte_carlo_rejects_invalid_strike() -> None:
    with pytest.raises(
        ValueError,
        match="strike must be positive",
    ):
        price_european_call_mc(
            spot=100.0,
            strike=0.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=100,
            seed=42,
        )


def test_monte_carlo_single_path_has_zero_standard_error() -> None:
    result = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=1,
        seed=42,
    )

    assert result.standard_error == 0.0
    assert result.n_paths == 1