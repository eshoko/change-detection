import pytest

from changedet.generator import generate_all_levels

VALIDATION_SEED = 2026
N_PER_LEVEL = 400


@pytest.fixture(scope="session")
def items():
    """Validation item set: 400 items per level, fixed seed."""
    return generate_all_levels(N_PER_LEVEL, VALIDATION_SEED)
