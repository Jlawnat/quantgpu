import pytest

from quantgpu.simulation.rng import MAX_SEED, validate_seed


def test_none_seed_is_allowed() -> None:
    assert validate_seed(None) is None


@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
        42,
        MAX_SEED,
    ],
)
def test_valid_seed_is_returned(seed: int) -> None:
    assert validate_seed(seed) == seed


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_seed(-1)


def test_seed_above_maximum_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_seed(MAX_SEED + 1)


@pytest.mark.parametrize(
    "seed",
    [
        True,
        False,
        1.5,
        "42",
    ],
)
def test_non_integer_seed_is_rejected(seed: object) -> None:
    with pytest.raises(TypeError):
        validate_seed(seed)  # type: ignore[arg-type]