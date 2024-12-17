import pickle
import random
from abc import ABC, abstractmethod
from typing import Any


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str | None:
        raise NotImplementedError


class SentenceRemoval(Augmentation):
    def __init__(self, start: int = 1, end: int = 1, k: int = 3, min_sentence_length: int = 10) -> None:
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

    def __str__(self) -> str:
        return "sentence-removal"


class SetenceShuffle(Augmentation):
    def __init__(self, window_size: int = 3, min_sentence_length: int = 10) -> None:
        self.window_size = window_size
        self.min_sentence_length = min_sentence_length

        assert (self.window_size * 2 + 1) < min_sentence_length - 2, (
            f"Can't create augmentor that shuffles {self.window_size * 2 + 1} sentences"
            f"for texts with minimum {self.min_sentence_length} sentences long"
            f"with window size = {self.window_size}"
        )

    def __call__(self, text: str) -> str | None:
        if text is None:
            return None
        sentences = [sentence.strip() for sentence in text.strip().split(".") if sentence.strip()]
        if len(sentences) < self.min_sentence_length:
            return None
        middle = len(sentences) // 2
        shuffle_block = sentences[middle - self.window_size : middle + self.window_size]
        random.shuffle(shuffle_block)
        return (
            ". ".join(sentences[: middle - self.window_size] + shuffle_block + sentences[middle + self.window_size :])
            + "."
        )

    def __str__(self) -> str:
        return "sentence-shuffle"


class KeyboardSwap(Augmentation):
    def __init__(
        self,
        graph_path: str = "./data/augmentation_models/hebrew_keyboard.gpickle",
        word_p: float = 0.5,
        char_p: float = 0.1,
        insert_double: bool = True,
    ) -> None:
        assert 0.0 <= word_p <= 1.0, "the word replacement probability can only be between 0 and 1"
        assert 0.0 <= char_p <= 1.0, "the character replacement probability can only be between 0 and 1"

        with open(graph_path, "rb") as fd:
            self.graph = pickle.load(fd)
        self.word_p = word_p
        self.char_p = char_p
        self.insert_double = insert_double

    def __call__(self, text: str) -> str | None:
        words: list[str] = text.split()
        for word_idx, word in enumerate(words):
            if random.random() < self.word_p:
                chars: list[str] = list(word)
                for char_idx, char in enumerate(chars):
                    if random.random() < self.char_p and char in self.graph:
                        random_char = random.sample(list(self.graph.neighbors(char)), k=1)[0]
                        chars[char_idx] = random_char + char * random.randint(0, 1) * self.insert_double
                words[word_idx] = "".join(chars)

        return " ".join(words)

    def __str__(self) -> str:
        return "keyboard-swapping"


def get_augmentations(augs_type: list[str], augmentations_config: dict[str, Any] | None) -> list[Augmentation]:
    augmentations: list[Augmentation] = []
    for type_ in augs_type:
        config = augmentations_config.get(type_, {}) if augmentations_config is not None else {}
        augmentation: Augmentation
        match type_:
            case "sentence-removal":
                augmentation = SentenceRemoval(**config)
            case "sentence-shuffle":
                augmentation = SetenceShuffle(**config)
            case "keyboard-swapping":
                augmentation = KeyboardSwap(**config)
            case _:
                raise ValueError(f"{type_} is not a valid augmentation")
        augmentations.append(augmentation)
    return augmentations
