# QuantGPU GPU Optimization Summary

This document records the GPU optimization path used for QuantGPU's
European-call Monte Carlo workload.

## Canonical workload

All headline GPU optimization results use:

- European call option pricing under GBM
- spot = 100
- strike = 100
- maturity = 1 year
- risk-free rate = 5%
- volatility = 20%
- 10,000,000 Monte Carlo paths
- seed = 42
- Tesla T4 GPU
- 3 warm-up runs
- 10 measured repetitions
- numerical validation against the Black–Scholes reference price

All candidates must pass the QuantGPU numerical correctness gate before
their timing results are accepted.

## Optimization progression

### PyTorch CUDA FP64

The original CUDA baseline used double precision.

Canonical end-to-end latency:

- 10.673 ms in the frozen CPU/GPU baseline benchmark

This established the reference GPU implementation for subsequent
optimization work.

### PyTorch CUDA FP32

Switching the Monte Carlo pipeline from FP64 to FP32 reduced execution
time substantially while remaining statistically consistent with the
Black–Scholes reference.

Final optimization-benchmark latency:

- 3.060 ms

This demonstrated that reduced precision was numerically admissible for
the current Monte Carlo workload.

### Reduced-intermediate FP32

A manually simplified FP32 pipeline was tested to reduce intermediate
tensor materialization.

Final latency:

- 3.063 ms

This produced no meaningful improvement over eager FP32 execution and
was therefore not selected as the final optimization path.

### torch.compile FP32

The payoff and reduction pipeline was compiled using `torch.compile`
with RNG setup kept outside the compiled graph.

Final latency:

- 1.429 ms

This was approximately:

- 2.14× faster than eager FP32
- 7.95× faster than FP64 in the same optimization benchmark

The compiled backend passed numerical correctness and reproducibility
validation.

### Triton CUDA FP32

A custom Triton kernel was implemented to fuse:

- GBM terminal transformation
- option payoff calculation
- discounting
- block payoff sum
- block squared-payoff sum

PyTorch remained responsible for Gaussian random-number generation,
while the custom kernel avoided materializing the full payoff pipeline
as separate tensors.

Final latency:

- 0.683 ms

Final throughput:

- approximately 14.83 billion paths per second

The Triton backend was approximately:

- 4.48× faster than eager FP32
- 2.09× faster than compiled FP32
- 16.63× faster than FP64 in the same optimization benchmark
- 15.63× faster than the separately frozen 10.673 ms FP64 baseline

The Triton implementation passed:

- Black–Scholes statistical validation
- Monte Carlo tolerance checks
- reproducibility checks
- standard-error validation

## Final backend decision

The selected optimized backend is:

`triton_cuda_fp32`

It provides the best validated performance among the implementations
tested while preserving the QuantGPU pricing-result contract and
numerical correctness requirements.

## Final optimization benchmark

The canonical Triton-inclusive benchmark is stored at:

`benchmarks/results/cuda_optimization_comparison_v2.csv`

The frozen CPU/GPU baseline is stored at:

`benchmarks/results/cpu_gpu_comparison_v2.csv`

These saved benchmark files are the source of truth for reported
performance results.