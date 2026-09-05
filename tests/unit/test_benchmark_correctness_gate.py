import pytest

from quantgpu.backends.protocol import PricingResult
from quantgpu.benchmarking.validation import require_valid_result


def test_benchmark_gate_accepts_valid_result() -> None:
    result = PricingResult(
        price=10.45,
        standard_error=0.02,
        n_paths=100_000,
    )

    require_valid_result(
        result,
        reference_price=10.4506,
    )


def test_benchmark_gate_rejects_invalid_result() -> None:
    result = PricingResult(
        price=12.0,
        standard_error=0.02,
        n_paths=100_000,
    )

    with pytest.raises(
        RuntimeError,
        match="Benchmark result failed numerical validation",
    ):
        require_valid_result(
            result,
            reference_price=10.4506,
        )