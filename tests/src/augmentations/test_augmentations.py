import pytest

from src.augmentations import Augmentation, get_augmentations


def test_augmentation():
    with pytest.raises(TypeError) as excinfo:
        _ = Augmentation()

    assert excinfo.type is TypeError


def test_sub_augmentation():
    class AugMock(Augmentation):
        def __call__(self, text: str) -> str | None:
            return super().__call__(text)  # type: ignore

    aug = AugMock()
    with pytest.raises(NotImplementedError) as excinfo:
        aug("")

    assert excinfo.type is NotImplementedError


def test_get_no_augmentations(empty_config):
    augmentations = get_augmentations([], augmentations_config=empty_config)
    assert augmentations == []


def test_get_invalid_augmentation(empty_config):
    with pytest.raises(ValueError) as excinfo:
        get_augmentations(["foobar"], augmentations_config=empty_config)
    assert excinfo.type is ValueError
