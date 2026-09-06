from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemInfo:
    """Basic hardware and software metadata for benchmark reproducibility."""

    os: str
    os_release: str
    machine: str
    processor: str
    python_version: str
    cpu_model: str


def _read_cpu_model() -> str:
    """Return the Linux CPU model when available."""
    try:
        result = subprocess.run(
            ["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"],
            check=True,
            capture_output=True,
            text=True,
        )
        cpu_model = result.stdout.strip()
        return cpu_model or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_system_info() -> SystemInfo:
    """Collect reproducibility metadata for the current machine."""
    return SystemInfo(
        os=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=platform.python_version(),
        cpu_model=_read_cpu_model(),
    )
