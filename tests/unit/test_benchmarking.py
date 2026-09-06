import subprocess

import pytest

from quantgpu.benchmarking import system_info, timer
from quantgpu.benchmarking.schema import BENCHMARK_SCHEMA_VERSION


def test_benchmark_schema_version_is_defined() -> None:
    assert BENCHMARK_SCHEMA_VERSION == "1.0"


def test_benchmark_callable_reports_timing_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter(
        [
            1.0,
            1.1,
            2.0,
            2.3,
            3.0,
            3.2,
        ]
    )

    monkeypatch.setattr(
        timer,
        "perf_counter",
        lambda: next(times),
    )

    calls = 0

    def function() -> None:
        nonlocal calls
        calls += 1

    result = timer.benchmark_callable(
        function,
        warmup_runs=1,
        repetitions=3,
    )

    assert calls == 4
    assert result.median_seconds == pytest.approx(0.2)
    assert result.min_seconds == pytest.approx(0.1)
    assert result.max_seconds == pytest.approx(0.3)
    assert result.repetitions == 3


def test_benchmark_callable_rejects_negative_warmup() -> None:
    with pytest.raises(
        ValueError,
        match="warmup_runs must be non-negative",
    ):
        timer.benchmark_callable(
            lambda: None,
            warmup_runs=-1,
        )


def test_benchmark_callable_rejects_non_positive_repetitions() -> None:
    with pytest.raises(
        ValueError,
        match="repetitions must be positive",
    ):
        timer.benchmark_callable(
            lambda: None,
            repetitions=0,
        )


def test_read_cpu_model_returns_detected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        stdout = "Intel Test CPU\n"

    monkeypatch.setattr(
        system_info.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )

    assert system_info._read_cpu_model() == "Intel Test CPU"


def test_read_cpu_model_returns_unknown_for_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        stdout = ""

    monkeypatch.setattr(
        system_info.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )

    assert system_info._read_cpu_model() == "unknown"


@pytest.mark.parametrize(
    "exception",
    [
        subprocess.CalledProcessError(
            returncode=1,
            cmd=["bash"],
        ),
        FileNotFoundError(),
    ],
)
def test_read_cpu_model_returns_unknown_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise exception

    monkeypatch.setattr(
        system_info.subprocess,
        "run",
        fail,
    )

    assert system_info._read_cpu_model() == "unknown"


def test_get_system_info_collects_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_info.platform,
        "system",
        lambda: "TestOS",
    )
    monkeypatch.setattr(
        system_info.platform,
        "release",
        lambda: "1.0",
    )
    monkeypatch.setattr(
        system_info.platform,
        "machine",
        lambda: "x86_64",
    )
    monkeypatch.setattr(
        system_info.platform,
        "processor",
        lambda: "Test Processor",
    )
    monkeypatch.setattr(
        system_info.platform,
        "python_version",
        lambda: "3.12.0",
    )
    monkeypatch.setattr(
        system_info,
        "_read_cpu_model",
        lambda: "Test CPU",
    )

    result = system_info.get_system_info()

    assert result.os == "TestOS"
    assert result.os_release == "1.0"
    assert result.machine == "x86_64"
    assert result.processor == "Test Processor"
    assert result.python_version == "3.12.0"
    assert result.cpu_model == "Test CPU"