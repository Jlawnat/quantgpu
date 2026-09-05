from __future__ import annotations

from math import sqrt

import numpy as np
from numpy.typing import NDArray


def simulate_gbm_terminal(
    *,
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate terminal prices under geometric Brownian motion."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    if maturity == 0:
        return np.full(n_paths, spot, dtype=np.float64)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    exponent = (
        (drift - 0.5 * volatility**2) * maturity
        + volatility * sqrt(maturity) * z
    )

    return spot * np.exp(exponent)