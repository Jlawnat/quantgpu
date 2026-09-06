from __future__ import annotations

MAX_SEED = 2**63 - 1


def validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    if seed > MAX_SEED:
        raise ValueError(f"seed must not exceed {MAX_SEED}")

    return seed
