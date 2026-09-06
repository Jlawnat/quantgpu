from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.pricing.monte_carlo import price_european_call_mc


def test_monte_carlo_error_decreases_with_more_paths() -> None:
    reference = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    small = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=10_000,
        seed=42,
    )

    large = price_european_call_mc(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=1_000_000,
        seed=42,
    )

    small_error = abs(small.price - reference)
    large_error = abs(large.price - reference)

    assert large_error < small_error
