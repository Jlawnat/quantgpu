from __future__ import annotations

from collections.abc import Mapping

BENCHMARK_SCHEMA_VERSION = "1.1"

REQUIRED_REPRODUCIBILITY_FIELDS = (
    "schema_version",
    "git_commit",
    "git_tree_state",
    "timestamp_utc",
    "backend",
    "device",
    "dtype",
    "python_version",
    "quantgpu_version",
    "numpy_version",
    "torch_version",
    "cuda_version",
    "triton_version",
    "n_paths",
    "warmup_runs",
    "repetitions",
    "seed",
)


def validate_benchmark_metadata(
    row: Mapping[str, object],
) -> None:
    """Validate required reproducibility metadata for a benchmark row."""
    missing = [
        field
        for field in REQUIRED_REPRODUCIBILITY_FIELDS
        if field not in row
    ]

    if missing:
        raise ValueError(
            "benchmark row missing required metadata: "
            + ", ".join(missing)
        )

    blank = [
        field
        for field in REQUIRED_REPRODUCIBILITY_FIELDS
        if row[field] is None or row[field] == ""
    ]

    if blank:
        raise ValueError(
            "benchmark row contains blank metadata: "
            + ", ".join(blank)
        )

    if row["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            "benchmark row schema version does not match "
            f"{BENCHMARK_SCHEMA_VERSION}"
        )

    if row["git_tree_state"] not in {
        "clean",
        "dirty",
        "unknown",
    }:
        raise ValueError(
            "git_tree_state must be clean, dirty, or unknown"
        )