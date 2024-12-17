import random

import pytest

from src.augmentations import Augmentation, SetenceShuffle, get_augmentations


@pytest.mark.parametrize("types", [["sentence-shuffle"]])
def test_get_sentence_shuffle_augmentation(types, empty_config):
    augmentations = get_augmentations(types, augmentations_config=empty_config)
    assert len(augmentations) == 1
    assert isinstance(augmentations[0], Augmentation)
    assert isinstance(augmentations[0], SetenceShuffle)


def test_shuffle_name():
    augmentation = SetenceShuffle()
    assert str(augmentation) == "sentence-shuffle"


def test_sentence_shuffle_text_is_none():
    augmentation = SetenceShuffle()
    text = None
    augmented_text = augmentation(text)
    assert augmented_text is None


def test_sentence_shuffle_text_is_short_than_10():
    augmentation = SetenceShuffle()
    text = "aaa. bbb. ccc."
    augmented_text = augmentation(text)
    assert augmented_text is None


def test_sentence_shuffle_valid_text():
    random.seed(2702)
    augmentation = SetenceShuffle(window_size=1)
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11."
    augmented_text = augmentation(text)
    assert len(augmented_text) == len(text)
    assert augmented_text == "1. 2. 3. 4. 6. 5. 7. 8. 9. 10. 11."


def test_sentence_no_edge_shuffle():
    augmentation = SetenceShuffle()
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    for i in range(100):
        augmented_text = augmentation(text)
        assert len(augmented_text) == len(text)
        assert augmented_text.startswith("1.")
        assert augmented_text.endswith("10.")


def test_min_sentence_length():
    augmentation = SetenceShuffle(min_sentence_length=30)
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    augmented_text = augmentation(text)
    assert augmented_text is None
