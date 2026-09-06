# QuantGPU Reference Environments

This directory stores exact software-environment snapshots associated with
canonical validation or benchmark runs.

Do not manually invent version snapshots.

Snapshots must be generated from the actual environment that successfully
passes the relevant QuantGPU validation gate.

Planned reference snapshots include:

- canonical CPU environment
- Tesla T4 GPU environment

Project compatibility requirements remain defined in `pyproject.toml`.

Environment snapshots are reproducibility artifacts, not replacements for the
project dependency specification.