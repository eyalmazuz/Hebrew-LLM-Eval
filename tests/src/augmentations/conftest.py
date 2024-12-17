import pytest


@pytest.fixture(scope="module")
def empty_config():
    return {}
