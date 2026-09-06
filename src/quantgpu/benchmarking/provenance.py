from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceProvenance:
    """Source-control metadata for a benchmark run."""

    git_commit: str
    git_tree_state: str


def _run_git_command(*args: str) -> str | None:
    """Run a Git command and return stripped stdout when available."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip()


def get_source_provenance() -> SourceProvenance:
    """Return the current Git commit and working-tree state."""
    commit = _run_git_command(
        "rev-parse",
        "HEAD",
    )

    status = _run_git_command(
        "status",
        "--porcelain",
    )

    if status is None:
        tree_state = "unknown"
    elif status:
        tree_state = "dirty"
    else:
        tree_state = "clean"

    return SourceProvenance(
        git_commit=commit or "unknown",
        git_tree_state=tree_state,
    )