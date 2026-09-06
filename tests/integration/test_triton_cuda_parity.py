import pytest
import torch

from quantgpu.pricing.black_scholes import black_scholes_call
from quantgpu.validation.tolerances import monte_carlo_tolerance

triton_backend = pytest.importorskip(
    "quantgpu.backends.triton_cuda",
    reason="Triton is not installed",
)

price_european_call_triton_cuda = (
    triton_backend.price_european_call_triton_cuda
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available",
    ),
]

def test_triton_cuda_matches_black_scholes() -> None:
    result = price_european_call_triton_cuda(
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


def test_triton_cuda_is_reproducible() -> None:
    first = price_european_call_triton_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
        dtype=torch.float32,
    )

    second = price_european_call_triton_cuda(
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


def test_triton_cuda_reports_standard_error() -> None:
    result = price_european_call_triton_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=100_000,
        seed=42,
        dtype=torch.float32,
    )

    assert result.standard_error > 0.0
    assert result.n_paths == 100_000