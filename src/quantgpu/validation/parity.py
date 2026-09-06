from __future__ import annotations

from dataclasses import dataclass

from quantgpu.backends.protocol import PricingResult
from quantgpu.validation.tolerances import (
    combined_monte_carlo_tolerance,
)


@dataclass(frozen=True)
class BackendComparison:
    absolute_difference: float
    tolerance: float
    within_tolerance: bool


def compare_pricing_results(
    first: PricingResult,
    second: PricingResult,
    *,
    z_score: float = 4.0,
) -> BackendComparison:
    difference = abs(first.price - second.price)

    tolerance = combined_monte_carlo_tolerance(
        first.standard_error,
        second.standard_error,
        z_score=z_score,
    )

    return BackendComparison(
        absolute_difference=difference,
        tolerance=tolerance,
        within_tolerance=difference <= tolerance,
    )
