from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class NumericalTolerance:
    absolute: float
    relative: float


FLOAT64_TOLERANCE = NumericalTolerance(
    absolute=1e-12,
    relative=1e-12,
)

FLOAT32_TOLERANCE = NumericalTolerance(
    absolute=1e-6,
    relative=1e-5,
)

MONTE_CARLO_Z_SCORE = 4.0


def monte_carlo_tolerance(
    standard_error: float,
    *,
    z_score: float = MONTE_CARLO_Z_SCORE,
) -> float:
    if standard_error < 0.0:
        raise ValueError("standard_error must be non-negative")

    if z_score <= 0.0:
        raise ValueError("z_score must be positive")

    return z_score * standard_error


def combined_monte_carlo_tolerance(
    first_standard_error: float,
    second_standard_error: float,
    *,
    z_score: float = MONTE_CARLO_Z_SCORE,
) -> float:
    if first_standard_error < 0.0:
        raise ValueError("first_standard_error must be non-negative")

    if second_standard_error < 0.0:
        raise ValueError("second_standard_error must be non-negative")

    if z_score <= 0.0:
        raise ValueError("z_score must be positive")

    combined_error = sqrt(
        first_standard_error**2
        + second_standard_error**2
    )

    return z_score * combined_error