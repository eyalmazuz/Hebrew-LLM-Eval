import random
from abc import ABC, abstractmethod
from typing import Any


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str | None:
        pass


class SentenceRemoval(Augmentation):
    def __init__(self, start: int = 1, end: int = 1, k: int = 2, min_sentence_length: int = 10) -> None:
        self.start = start
        self.end = end
        self.k = k
        self.min_sentence_length = min_sentence_length

        assert self.min_sentence_length >= self.k + self.start + self.end, (
            f"Can't create augmentor that removes {self.k} sentences"
            f"for texts with minimum {self.min_sentence_length} sentences long"
            f"with start = {self.start} and end = {self.end}"
        )

    def __call__(self, text: str) -> str | None:
        if text is None:
            return None
        sentences = [sentence.strip() for sentence in text.strip().split(".") if sentence.strip()]
        if len(sentences) < self.min_sentence_length:
            return None
        indices_to_remove = random.sample(range(self.start, len(sentences) - self.end), k=self.k)
        augmented_sentences = [sentence for i, sentence in enumerate(sentences) if i not in indices_to_remove]
        return ". ".join(augmented_sentences) + "."


def get_augmentations(augs_type: list[str], augmentations_config: dict[str, Any] | None) -> list[Augmentation]:
    augmentations: list[Augmentation] = []
    for type_ in augs_type:
        config = augmentations_config.get(type_, {}) if augmentations_config is not None else {}
        match type_:
            case "sentence-removal":
                augmentation = SentenceRemoval(**config)
            case _:
                raise ValueError(f"{type_} is not a valid augmentation")
        augmentations.append(augmentation)
    return augmentations
