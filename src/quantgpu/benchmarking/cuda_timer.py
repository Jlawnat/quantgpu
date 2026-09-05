from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any, cast

import torch


@dataclass(frozen=True)
class CudaTimingResult:
    """Summary statistics for repeated CUDA timings."""

    median_seconds: float
    min_seconds: float
    max_seconds: float
    repetitions: int


def benchmark_cuda_callable[T](
    function: Callable[[], T],
    *,
    warmup_runs: int = 3,
    repetitions: int = 10,
) -> CudaTimingResult:
    """Benchmark a CUDA callable using synchronized CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    for _ in range(warmup_runs):
        function()

    torch.cuda.synchronize()

    timings: list[float] = []
    event_factory = cast(Any, torch.cuda.Event)

    for _ in range(repetitions):
        start = event_factory(enable_timing=True)
        end = event_factory(enable_timing=True)
        
        start.record()
        function()
        end.record()

        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        timings.append(elapsed_ms / 1_000.0)

    return CudaTimingResult(
        median_seconds=median(timings),
        min_seconds=min(timings),
        max_seconds=max(timings),
        repetitions=repetitions,
    )