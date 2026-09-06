import pytest

from quantgpu.benchmarking.schema import (
    BENCHMARK_SCHEMA_VERSION,
    REQUIRED_REPRODUCIBILITY_FIELDS,
    validate_benchmark_metadata,
)


def _valid_row() -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": "abc123",
        "git_tree_state": "clean",
        "timestamp_utc": "2026-09-06T00:00:00+00:00",
        "backend": "numpy_cpu",
        "device": "cpu",
        "dtype": "float64",
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "n_paths": 10_000_000,
        "seed": 42,
        "warmup_runs": 1,
        "repetitions": 5,
        "python_version": "3.12.0",
        "quantgpu_version": "0.1.0",
        "numpy_version": "2.0.0",
        "torch_version": "2.10.0",
        "cuda_version": "none",
        "triton_version": "not-installed",
        "os": "Linux",
        "cpu_model": "Test CPU",
        "validation_status": "passed",
    }


def test_required_reproducibility_fields_are_defined() -> None:
    assert "git_commit" in REQUIRED_REPRODUCIBILITY_FIELDS
    assert "git_tree_state" in REQUIRED_REPRODUCIBILITY_FIELDS
    assert "dtype" in REQUIRED_REPRODUCIBILITY_FIELDS
    assert "quantgpu_version" in REQUIRED_REPRODUCIBILITY_FIELDS
    assert "seed" in REQUIRED_REPRODUCIBILITY_FIELDS


def test_valid_benchmark_metadata_is_accepted() -> None:
    validate_benchmark_metadata(_valid_row())


def test_missing_metadata_is_rejected() -> None:
    row = _valid_row()
    del row["git_commit"]

    with pytest.raises(
        ValueError,
        match="missing required metadata",
    ):
        validate_benchmark_metadata(row)


def test_blank_metadata_is_rejected() -> None:
    row = _valid_row()
    row["torch_version"] = ""

    with pytest.raises(
        ValueError,
        match="blank metadata",
    ):
        validate_benchmark_metadata(row)


def test_wrong_schema_version_is_rejected() -> None:
    row = _valid_row()
    row["schema_version"] = "999.0"

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        validate_benchmark_metadata(row)


def test_invalid_git_tree_state_is_rejected() -> None:
    row = _valid_row()
    row["git_tree_state"] = "maybe"

    with pytest.raises(
        ValueError,
        match="git_tree_state",
    ):
        validate_benchmark_metadata(row)
def test_failed_validation_status_is_rejected() -> None:
    row = _valid_row()
    row["validation_status"] = "failed"

    with pytest.raises(
        ValueError,
        match="must pass numerical validation",
    ):
        validate_benchmark_metadata(row)


def test_cuda_metadata_requires_gpu_name() -> None:
    row = _valid_row()
    row["device"] = "cuda"
    row["cuda_version"] = "12.8"

    with pytest.raises(
        ValueError,
        match="must include gpu_name",
    ):
        validate_benchmark_metadata(row)

    row["gpu_name"] = "Tesla T4"

    validate_benchmark_metadata(row)

