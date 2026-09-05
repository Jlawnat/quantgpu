import pytest
import torch

from quantgpu.backends.numpy_cpu import price_european_call_numpy_cpu
from quantgpu.backends.torch_cpu import price_european_call_torch_cpu
from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.backends.torch_cuda_fused import (
    price_european_call_torch_cuda_fused,
)
from quantgpu.validation.parity import compare_pricing_results


def test_numpy_and_torch_cpu_are_statistically_consistent() -> None:
    numpy_result = price_european_call_numpy_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    torch_result = price_european_call_torch_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    comparison = compare_pricing_results(
        numpy_result,
        torch_result,
    )

    assert comparison.within_tolerance


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_numpy_and_torch_cuda_are_statistically_consistent() -> None:
    numpy_result = price_european_call_numpy_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    cuda_result = price_european_call_torch_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    comparison = compare_pricing_results(
        numpy_result,
        cuda_result,
    )

    assert comparison.within_tolerance


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_torch_cpu_and_cuda_are_statistically_consistent() -> None:
    cpu_result = price_european_call_torch_cpu(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    cuda_result = price_european_call_torch_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
    )

    comparison = compare_pricing_results(
        cpu_result,
        cuda_result,
    )

    assert comparison.within_tolerance


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_fp64_and_fused_fp32_are_statistically_consistent() -> None:
    fp64_result = price_european_call_torch_cuda(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
        dtype=torch.float64,
    )

    fp32_result = price_european_call_torch_cuda_fused(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        n_paths=500_000,
        seed=42,
        dtype=torch.float32,
    )

    comparison = compare_pricing_results(
        fp64_result,
        fp32_result,
    )

    assert comparison.within_tolerance