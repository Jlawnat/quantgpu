import numpy as np
import pytest

from quantgpu.simulation.gbm import simulate_gbm_terminal


def test_gbm_empirical_variance_matches_theory() -> None:
    spot = 100.0
    drift = 0.05
    volatility = 0.20
    maturity = 1.0

    paths = simulate_gbm_terminal(
        spot=spot,
        drift=drift,
        volatility=volatility,
        maturity=maturity,
        n_paths=300_000,
        seed=42,
    )

    empirical_variance = float(np.var(paths, ddof=1))

    theoretical_variance = (
        spot**2
        * np.exp(2.0 * drift * maturity)
        * (np.exp(volatility**2 * maturity) - 1.0)
    )

    assert empirical_variance == pytest.approx(
        theoretical_variance,
        rel=0.03,
    )