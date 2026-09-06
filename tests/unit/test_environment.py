import builtins
import importlib.metadata

import torch

from quantgpu import __version__
from quantgpu.benchmarking import environment


def test_project_version() -> None:
    assert __version__ == "0.1.0"


def test_distribution_version_returns_installed_version(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        environment.importlib.metadata,
        "version",
        lambda name: "1.2.3",
    )

    assert environment._distribution_version("example") == "1.2.3"


def test_distribution_version_handles_missing_package(
    monkeypatch,
) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        environment.importlib.metadata,
        "version",
        missing,
    )

    assert environment._distribution_version("missing-package") == "not-installed"


def test_get_software_environment(
    monkeypatch,
) -> None:
    versions = {
        "torch": "2.10.0",
        "triton": "3.6.0",
    }

    monkeypatch.setattr(
        environment,
        "_distribution_version",
        lambda name: versions[name],
    )

    monkeypatch.setattr(
        environment,
        "_cuda_version",
        lambda: "12.8",
    )

    monkeypatch.setattr(
        environment.platform,
        "python_version",
        lambda: "3.12.13",
    )

    result = environment.get_software_environment()

    assert result.python_version == "3.12.13"
    assert result.quantgpu_version == "0.1.0"
    assert result.numpy_version
    assert result.torch_version == "2.10.0"
    assert result.cuda_version == "12.8"
    assert result.triton_version == "3.6.0"


def test_cuda_version_with_torch_available() -> None:
    expected = torch.version.cuda or "none"

    assert environment._cuda_version() == expected


def test_cuda_version_handles_missing_torch(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name == "torch":
            raise ImportError("torch is unavailable")

        return real_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        fake_import,
    )

    assert environment._cuda_version() == "not-available"
