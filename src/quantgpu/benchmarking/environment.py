from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass

import numpy as np

from quantgpu import __version__


@dataclass(frozen=True)
class SoftwareEnvironment:
    """Software-version metadata for reproducible benchmark runs."""

    python_version: str
    quantgpu_version: str
    numpy_version: str
    torch_version: str
    cuda_version: str
    triton_version: str


def _distribution_version(name: str) -> str:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cuda_version() -> str:
    """Return the CUDA version reported by PyTorch when available."""
    try:
        import torch
    except ImportError:
        return "not-available"

    return torch.version.cuda or "none"


def get_software_environment() -> SoftwareEnvironment:
    """Collect software metadata for benchmark provenance."""
    return SoftwareEnvironment(
        python_version=platform.python_version(),
        quantgpu_version=__version__,
        numpy_version=np.__version__,
        torch_version=_distribution_version("torch"),
        cuda_version=_cuda_version(),
        triton_version=_distribution_version("triton"),
    )