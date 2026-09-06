from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from quantgpu.backends.protocol import PricingResult
from quantgpu.backends.torch_cuda import price_european_call_torch_cuda
from quantgpu.backends.torch_cuda_compiled import (
    price_european_call_torch_cuda_compiled,
)
from quantgpu.backends.torch_cuda_fused import (
    price_european_call_torch_cuda_fused,
)
from quantgpu.backends.triton_cuda import (
    price_european_call_triton_cuda,
)
from quantgpu.benchmarking.cuda_timer import benchmark_cuda_callable
from quantgpu.benchmarking.environment import get_software_environment
from quantgpu.benchmarking.provenance import get_source_provenance
from quantgpu.benchmarking.schema import BENCHMARK_SCHEMA_VERSION
from quantgpu.benchmarking.system_info import get_system_info
from quantgpu.benchmarking.validation import require_valid_result
from quantgpu.pricing.black_scholes import black_scholes_call

RESULTS_DIR = Path("benchmarks/results")
RESULTS_FILE = RESULTS_DIR / "cuda_optimization_comparison_v3.csv"

N_PATHS = 10_000_000
WARMUP_RUNS = 3
REPETITIONS = 10
PRECONDITION_RUNS = 5
SEED = 42

BackendFunction = Callable[..., PricingResult]


def _precondition_gpu() -> None:
    for _ in range(PRECONDITION_RUNS):
        price_european_call_torch_cuda(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=SEED,
            dtype=torch.float64,
        )

    torch.cuda.synchronize()


def _benchmark_candidate(
    *,
    name: str,
    backend: BackendFunction,
    dtype: torch.dtype,
    reference_price: float,
) -> dict[str, str | int | float]:
    def workload() -> PricingResult:
        return backend(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=SEED,
            dtype=dtype,
        )

    device_timing = benchmark_cuda_callable(
        workload,
        warmup_runs=WARMUP_RUNS,
        repetitions=REPETITIONS,
    )

    for _ in range(WARMUP_RUNS):
        workload()

    torch.cuda.synchronize()

    wall_times: list[float] = []

    for _ in range(REPETITIONS):
        start = perf_counter()

        workload()
        torch.cuda.synchronize()

        wall_times.append(perf_counter() - start)

    result = workload()
    require_valid_result(result, reference_price)

    wall_median_seconds = median(wall_times)
    system_info = get_system_info()
    provenance = get_source_provenance()
    software = get_software_environment()

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": provenance.git_commit,
        "git_tree_state": provenance.git_tree_state,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": name,
        "device": "cuda",
        "dtype": str(dtype),
        "python_version": software.python_version,
        "quantgpu_version": software.quantgpu_version,
        "numpy_version": software.numpy_version,
        "torch_version": software.torch_version,
        "cuda_version": software.cuda_version,
        "triton_version": software.triton_version,
        "gpu_name": torch.cuda.get_device_name(0),
        "os": system_info.os,
        "os_release": system_info.os_release,
        "machine": system_info.machine,
        "processor": system_info.processor,
        "cpu_model": system_info.cpu_model,
        "n_paths": N_PATHS,
        "warmup_runs": WARMUP_RUNS,
        "repetitions": REPETITIONS,
        "device_median_ms": device_timing.median_seconds * 1_000.0,
        "end_to_end_median_ms": wall_median_seconds * 1_000.0,
        "min_ms": device_timing.min_seconds * 1_000.0,
        "max_ms": device_timing.max_seconds * 1_000.0,
        "throughput_paths_per_sec": (
            N_PATHS / device_timing.median_seconds
        ),
        "estimated_price": result.price,
        "reference_price": reference_price,
        "absolute_error": abs(result.price - reference_price),
        "standard_error": result.standard_error,
        "seed": SEED,
    }


def _add_speedups(
    rows: list[dict[str, str | int | float]],
) -> None:
    fp64_row = next(
        row
        for row in rows
        if row["backend"] == "torch_cuda_fp64"
    )

    eager_fp32_row = next(
        row
        for row in rows
        if row["backend"] == "torch_cuda_fp32"
    )

    compiled_fp32_row = next(
        row
        for row in rows
        if row["backend"] == "torch_cuda_compiled_fp32"
    )

    fp64_ms = float(
        fp64_row["end_to_end_median_ms"]
    )

    eager_fp32_ms = float(
        eager_fp32_row["end_to_end_median_ms"]
    )

    compiled_fp32_ms = float(
        compiled_fp32_row["end_to_end_median_ms"]
    )

    for row in rows:
        row_ms = float(
            row["end_to_end_median_ms"]
        )

        row["speedup_vs_fp64"] = (
            fp64_ms / row_ms
        )

        row["speedup_vs_eager_fp32"] = (
            eager_fp32_ms / row_ms
        )

        row["speedup_vs_compiled_fp32"] = (
            compiled_fp32_ms / row_ms
        )


def _save_results(
    rows: list[dict[str, str | int | float]],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    _precondition_gpu()

    reference_price = black_scholes_call(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )

    candidates = [
        (
            "torch_cuda_fp64",
            price_european_call_torch_cuda,
            torch.float64,
        ),
        (
            "torch_cuda_fp32",
            price_european_call_torch_cuda,
            torch.float32,
        ),
        (
            "torch_cuda_fused_fp32",
            price_european_call_torch_cuda_fused,
            torch.float32,
        ),
        (
            "torch_cuda_compiled_fp32",
            price_european_call_torch_cuda_compiled,
            torch.float32,
        ),
        (
            "triton_cuda_fp32",
            price_european_call_triton_cuda,
            torch.float32,
        ),
    ]

    rows = [
        _benchmark_candidate(
            name=name,
            backend=backend,
            dtype=dtype,
            reference_price=reference_price,
        )
        for name, backend, dtype in candidates
    ]

    _add_speedups(rows)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Paths: {N_PATHS:,}")
    print()

    print(
        f"{'backend':<28}"
        f"{'device_ms':>12}"
        f"{'e2e_ms':>12}"
        f"{'paths/sec':>16}"
        f"{'vs_fp64':>10}"
        f"{'vs_fp32':>10}"
        f"{'vs_comp':>10}"
        f"{'abs_error':>12}"
    )

    for row in rows:
        print(
            f"{str(row['backend']):<28}"
            f"{float(row['device_median_ms']):12.3f}"
            f"{float(row['end_to_end_median_ms']):12.3f}"
            f"{float(row['throughput_paths_per_sec']):16,.0f}"
            f"{float(row['speedup_vs_fp64']):10.2f}"
            f"{float(row['speedup_vs_eager_fp32']):10.2f}"
            f"{float(row['speedup_vs_compiled_fp32']):10.2f}"
            f"{float(row['absolute_error']):12.6f}"
        )

    _save_results(rows)

    print(
        f"\nSaved results to {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()