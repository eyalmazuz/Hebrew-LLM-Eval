import random

import pytest

from src.augmentations import Augmentation, SentenceRemoval, get_augmentations


@pytest.mark.parametrize("types", [["sentence-removal"]])
def test_get_sentence_removal_augmentation(types, empty_config):
    augmentations = get_augmentations(types, augmentations_config=empty_config)
    assert len(augmentations) == 1
    assert isinstance(augmentations[0], Augmentation)
    assert isinstance(augmentations[0], SentenceRemoval)


def test_sentence_removal_text_is_none():
    augmentation = SentenceRemoval()
    text = None
    augmented_text = augmentation(text)
    assert augmented_text is None


def test_shuffle_name():
    augmentation = SentenceRemoval()
    assert str(augmentation) == "sentence-removal"


def test_sentence_removal_text_is_short_than_10():
    augmentation = SentenceRemoval()
    text = "aaa. bbb. ccc."
    augmented_text = augmentation(text)
    assert augmented_text is None


def test_sentence_removal_valid_text():
    random.seed(3009)
    augmentation = SentenceRemoval()
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    augmented_text = augmentation(text)
    assert len([sentence.strip() for sentence in augmented_text.strip().split(".") if sentence.strip()]) == 7
    assert augmented_text == "1. 2. 3. 5. 7. 8. 10."


def test_sentence_no_edge_removal():
    augmentation = SentenceRemoval()
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    for i in range(100):
        augmented_text = augmentation(text)
        assert len([sentence.strip() for sentence in augmented_text.strip().split(".") if sentence.strip()]) == 7
        assert augmented_text.startswith("1.")
        assert augmented_text.endswith("10.")


def test_sentence_only_edge_left():
    augmentation = SentenceRemoval(k=8)
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    augmented_text = augmentation(text)
    assert augmented_text == "1. 10."


def test_sentence_only_6_removed():
    augmentation = SentenceRemoval(start=4, end=4, k=2)
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    augmented_text = augmentation(text)
    assert augmented_text == "1. 2. 3. 4. 7. 8. 9. 10."


def test_min_sentence_length():
    augmentation = SentenceRemoval(min_sentence_length=30)
    text = "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    augmented_text = augmentation(text)
    assert augmented_text is None
