import pytest

from quantgpu.validation.tolerances import (
    FLOAT32_TOLERANCE,
    FLOAT64_TOLERANCE,
    combined_monte_carlo_tolerance,
    monte_carlo_tolerance,
)


def test_float64_tolerance_is_tighter_than_float32() -> None:
    assert FLOAT64_TOLERANCE.absolute < FLOAT32_TOLERANCE.absolute
    assert FLOAT64_TOLERANCE.relative < FLOAT32_TOLERANCE.relative


def test_monte_carlo_tolerance_uses_default_z_score() -> None:
    tolerance = monte_carlo_tolerance(0.02)

    assert tolerance == pytest.approx(0.08)


def test_monte_carlo_tolerance_accepts_custom_z_score() -> None:
    tolerance = monte_carlo_tolerance(
        0.02,
        z_score=3.0,
    )

    assert tolerance == pytest.approx(0.06)


def test_combined_monte_carlo_tolerance_uses_quadrature() -> None:
    tolerance = combined_monte_carlo_tolerance(
        0.03,
        0.04,
        z_score=2.0,
    )

    assert tolerance == pytest.approx(0.10)


def test_monte_carlo_tolerance_rejects_negative_standard_error() -> None:
    with pytest.raises(ValueError):
        monte_carlo_tolerance(-0.01)


def test_monte_carlo_tolerance_rejects_non_positive_z_score() -> None:
    with pytest.raises(ValueError):
        monte_carlo_tolerance(
            0.01,
            z_score=0.0,
        )


@pytest.mark.parametrize(
    ("first_standard_error", "second_standard_error"),
    [
        (-0.01, 0.02),
        (0.01, -0.02),
    ],
)
def test_combined_tolerance_rejects_negative_standard_error(
    first_standard_error: float,
    second_standard_error: float,
) -> None:
    with pytest.raises(ValueError):
        combined_monte_carlo_tolerance(
            first_standard_error,
            second_standard_error,
        )