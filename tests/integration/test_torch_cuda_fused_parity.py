import pytest
import torch

from quantgpu.backends.torch_cuda_fused import (
    price_european_call_torch_cuda_fused,
)
from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.validation.tolerances import monte_carlo_tolerance

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available",
    ),
]


def test_fused_cuda_matches_black_scholes() -> None:
    result = price_european_call_torch_cuda_fused(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
        dtype=torch.float32,
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


def test_fused_cuda_reproducible() -> None:
    first = price_european_call_torch_cuda_fused(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
        dtype=torch.float32,
    )

    second = price_european_call_torch_cuda_fused(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
        dtype=torch.float32,
    )

    assert first == second
