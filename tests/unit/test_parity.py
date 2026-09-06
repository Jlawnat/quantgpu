import pytest

from quantgpu.backends.protocol import PricingResult
from quantgpu.validation.parity import compare_pricing_results


def test_comparison_accepts_results_within_tolerance() -> None:
    first = PricingResult(
        price=10.45,
        standard_error=0.02,
        n_paths=100_000,
    )

    second = PricingResult(
        price=10.48,
        standard_error=0.02,
        n_paths=100_000,
    )

    comparison = compare_pricing_results(first, second)

    assert comparison.within_tolerance
    assert comparison.absolute_difference == pytest.approx(0.03)


def test_comparison_rejects_results_outside_tolerance() -> None:
    first = PricingResult(
        price=10.00,
        standard_error=0.01,
        n_paths=100_000,
    )

    second = PricingResult(
        price=11.00,
        standard_error=0.01,
        n_paths=100_000,
    )

    comparison = compare_pricing_results(first, second)

    assert not comparison.within_tolerance


def test_comparison_reports_expected_tolerance() -> None:
    first = PricingResult(
        price=10.0,
        standard_error=0.03,
        n_paths=100_000,
    )

    second = PricingResult(
        price=10.0,
        standard_error=0.04,
        n_paths=100_000,
    )

    comparison = compare_pricing_results(
        first,
        second,
        z_score=2.0,
    )

    assert comparison.tolerance == pytest.approx(0.10)
