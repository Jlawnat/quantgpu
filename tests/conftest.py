from pathlib import Path

import pytest


def pytest_collection_modifyitems(
    items: list[pytest.Item],
) -> None:
    """Apply test-tier markers from the test directory structure."""

    for item in items:
        path = Path(str(item.path))

        if "unit" in path.parts:
            item.add_marker(pytest.mark.unit)

        elif "integration" in path.parts:
            item.add_marker(pytest.mark.integration)

        elif "numerical" in path.parts:
            item.add_marker(pytest.mark.numerical)