import subprocess

from quantgpu.benchmarking import provenance


def test_source_provenance_reports_clean_commit(
    monkeypatch,
) -> None:
    outputs = iter(
        [
            "abc123\n",
            "",
        ]
    )

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Result(next(outputs))

    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        fake_run,
    )

    result = provenance.get_source_provenance()

    assert result.git_commit == "abc123"
    assert result.git_tree_state == "clean"


def test_source_provenance_reports_dirty_tree(
    monkeypatch,
) -> None:
    outputs = iter(
        [
            "abc123\n",
            " M src/example.py\n",
        ]
    )

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Result(next(outputs))

    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        fake_run,
    )

    result = provenance.get_source_provenance()

    assert result.git_commit == "abc123"
    assert result.git_tree_state == "dirty"


def test_source_provenance_handles_missing_git(
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        fail,
    )

    result = provenance.get_source_provenance()

    assert result.git_commit == "unknown"
    assert result.git_tree_state == "unknown"


def test_git_command_handles_command_failure(
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["git"],
        )

    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        fail,
    )

    assert provenance._run_git_command(
        "rev-parse",
        "HEAD",
    ) is None