from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PricingResult:
    """Common result returned by Monte Carlo pricing backends."""

    price: float
    standard_error: float
    n_paths: int


class EuropeanCallBackend(Protocol):
    """Protocol for European call Monte Carlo pricing backends."""

    def __call__(
        self,
        *,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        n_paths: int,
        seed: int | None = None,
    ) -> PricingResult:
        """Price a European call option."""
        ...