from __future__ import annotations

from typing import Protocol

from quantgpu.validation.tolerances import monte_carlo_tolerance


class PricingEstimate(Protocol):
    """Minimal pricing result required by the benchmark correctness gate."""

    price: float
    standard_error: float


def require_valid_result(
    result: PricingEstimate,
    reference_price: float,
) -> None:
    """Reject benchmark results that fail the Monte Carlo correctness gate."""
    tolerance = monte_carlo_tolerance(result.standard_error)

    if abs(result.price - reference_price) > tolerance:
        raise RuntimeError(
            "Benchmark result failed numerical validation: "
            f"price={result.price:.8f}, "
            f"reference={reference_price:.8f}, "
            f"tolerance={tolerance:.8f}"
        )