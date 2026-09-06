from __future__ import annotations

import torch

from quantgpu.backends.torch_cuda import price_european_call_torch_cuda

N_PATHS = 10_000_000


def main() -> None:
    """Profile CUDA operators for the FP32 Monte Carlo backend."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Warm up first so startup effects do not dominate the profile.
    for _ in range(3):
        price_european_call_torch_cuda(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=42,
            dtype=torch.float32,
        )

    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        price_european_call_torch_cuda(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            n_paths=N_PATHS,
            seed=42,
            dtype=torch.float32,
        )

        torch.cuda.synchronize()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Paths: {N_PATHS:,}")
    print("Dtype: torch.float32")
    print()

    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=25,
        )
    )


if __name__ == "__main__":
    main()
