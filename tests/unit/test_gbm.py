import numpy as np
import pytest

from quantgpu.simulation.gbm import simulate_gbm_terminal


def test_gbm_zero_maturity_returns_spot() -> None:
    paths = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.2,
        maturity=0.0,
        n_paths=1000,
        seed=42,
    )

    assert np.all(paths == 100.0)


def test_gbm_reproducibility_with_same_seed() -> None:
    first = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.2,
        maturity=1.0,
        n_paths=1000,
        seed=42,
    )

    second = simulate_gbm_terminal(
        spot=100.0,
        drift=0.05,
        volatility=0.2,
        maturity=1.0,
        n_paths=1000,
        seed=42,
    )

    assert np.array_equal(first, second)


def test_gbm_empirical_mean_matches_theory() -> None:
    spot = 100.0
    drift = 0.05
    maturity = 1.0

    paths = simulate_gbm_terminal(
        spot=spot,
        drift=drift,
        volatility=0.2,
        maturity=maturity,
        n_paths=200_000,
        seed=42,
    )

    theoretical_mean = spot * np.exp(drift * maturity)
    empirical_mean = float(np.mean(paths))

    assert empirical_mean == pytest.approx(theoretical_mean, rel=0.01)


@pytest.mark.parametrize(
    ("spot", "volatility", "maturity", "n_paths"),
    [
        (0.0, 0.2, 1.0, 100),
        (-1.0, 0.2, 1.0, 100),
        (100.0, -0.2, 1.0, 100),
        (100.0, 0.2, -1.0, 100),
        (100.0, 0.2, 1.0, 0),
    ],
)
def test_invalid_gbm_inputs_raise(
    spot: float,
    volatility: float,
    maturity: float,
    n_paths: int,
) -> None:
    with pytest.raises(ValueError):
        simulate_gbm_terminal(
            spot=spot,
            drift=0.05,
            volatility=volatility,
            maturity=maturity,
            n_paths=n_paths,
            seed=42,
        )
