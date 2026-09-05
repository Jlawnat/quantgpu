import pytest

from quantgpu.backends.torch_cpu import (
    price_european_call_torch_cpu,
)
from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.validation.tolerances import monte_carlo_tolerance

def test_torch_cpu_matches_black_scholes() -> None:
    result = price_european_call_torch_cpu(
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
        abs=monte_carlo_tolerance(result.standard_error),
    )


def test_torch_cpu_reproducibility() -> None:
    first = price_european_call_torch_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
    )

    second = price_european_call_torch_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
    )

    assert first == second


def test_torch_cpu_reports_standard_error() -> None:
    result = price_european_call_torch_cpu(
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