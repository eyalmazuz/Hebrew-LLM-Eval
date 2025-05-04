import pytest

from src.utils import load_data


def test_load_fails():
    with pytest.raises(FileNotFoundError):
        load_data("sdadasdag")

