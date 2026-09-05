from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter


@dataclass(frozen=True)
class TimingResult:
    """Summary statistics for repeated benchmark timings."""

    median_seconds: float
    min_seconds: float
    max_seconds: float
    repetitions: int


def benchmark_callable[T](
    function: Callable[[], T],
    *,
    warmup_runs: int = 1,
    repetitions: int = 5,
) -> TimingResult:
    """Benchmark a callable after optional warm-up runs."""
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    for _ in range(warmup_runs):
        function()

    timings: list[float] = []

    for _ in range(repetitions):
        start = perf_counter()
        function()
        end = perf_counter()

        timings.append(end - start)

    return TimingResult(
        median_seconds=median(timings),
        min_seconds=min(timings),
        max_seconds=max(timings),
        repetitions=repetitions,
    )