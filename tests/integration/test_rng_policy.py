import numpy as np
import torch

from quantgpu.backends.torch_cpu import simulate_gbm_terminal_torch_cpu
from quantgpu.simulation.gbm import simulate_gbm_terminal


def test_numpy_different_seeds_produce_different_paths() -> None:
    first = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=1,
    )

    second = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=2,
    )

    assert not np.array_equal(first, second)


def test_numpy_none_seed_produces_fresh_streams() -> None:
    first = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=None,
    )

    second = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=None,
    )

    assert not np.array_equal(first, second)


def test_torch_cpu_different_seeds_produce_different_paths() -> None:
    first = simulate_gbm_terminal_torch_cpu(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=1,
    )

    second = simulate_gbm_terminal_torch_cpu(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=2,
    )

    assert not torch.equal(first, second)


def test_torch_cpu_none_seed_produces_fresh_streams() -> None:
    first = simulate_gbm_terminal_torch_cpu(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=None,
    )

    second = simulate_gbm_terminal_torch_cpu(
        spot=100.0,
        drift=0.05,
        volatility=0.20,
        maturity=1.0,
        n_paths=1000,
        seed=None,
    )

    assert not torch.equal(first, second)