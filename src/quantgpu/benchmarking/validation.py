from __future__ import annotations

from quantgpu.backends.protocol import PricingResult
from quantgpu.validation.tolerances import monte_carlo_tolerance


def require_valid_result(
    result: PricingResult,
    reference_price: float,
) -> None:
    tolerance = monte_carlo_tolerance(result.standard_error)

    if abs(result.price - reference_price) > tolerance:
        raise RuntimeError(
            "Benchmark result failed numerical validation: "
            f"price={result.price:.8f}, "
            f"reference={reference_price:.8f}, "
            f"tolerance={tolerance:.8f}"
        )
