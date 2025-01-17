import pickle
import random
from abc import ABC, abstractmethod
from typing import Any

import fasttext
import fasttext.util


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, text: str) -> str | None:
        raise NotImplementedError


class BlockRemoval(Augmentation):
    def __init__(
        self,
        start: int = 1,
        end: int = 1,
        k: int = 3,
        min_sentence_length: int = 10,
        type_: str = "sentence",
        span_length: int = 3,
    ) -> None:
        self.start = start
        self.end = end
        self.k = k
        self.min_sentence_length = min_sentence_length
        self.type_ = type_
        self.span_length = span_length

        if self.type_ == "sentence":
            assert self.min_sentence_length >= self.k + self.start + self.end, (
                f"Can't create augmentor that removes {self.k} sentences"
                f"for texts with minimum {self.min_sentence_length} sentences long"
                f"with start = {self.start} and end = {self.end}"
            )

        assert self.type_ in ["sentence", "word", "span"], f"Invalid type {self.type_} was chose"

    def __call__(self, text: str) -> str | None:
        if text is None:
            return None
        if self.type_ == "sentence":
            sentences = [sentence.strip() for sentence in text.strip().split(".") if sentence.strip()]
            if len(sentences) < self.min_sentence_length:
                return None
            indices_to_remove = random.sample(range(self.start, len(sentences) - self.end), k=self.k)
            augmented_sentences = [sentence for i, sentence in enumerate(sentences) if i not in indices_to_remove]
            return ". ".join(augmented_sentences) + "."
        elif self.type_ == "word":
            sentences = [sentence.strip() for sentence in text.strip().split(".") if sentence.strip()]
            start_sents_len = len(" ".join(sentences[: self.start]).split(" "))
            end_sents_len = len(" ".join(sentences[-self.end :]).split(" "))
            if (len(text.split(" ")) - end_sents_len - start_sents_len) < self.k:
                return None
            indices_to_remove = random.sample(range(start_sents_len, len(text.split(" ")) - end_sents_len), k=self.k)
            augmented_words = [word for i, word in enumerate(text.split(" ")) if i not in indices_to_remove]
            return " ".join(augmented_words)
        elif self.type_ == "span":
            left_to_remove = self.k
            text_copy = text[:]
            sentences = [sentence.strip() for sentence in text_copy.strip().split(".") if sentence.strip()]
            start_sents_len = len(" ".join(sentences[: self.start]).split(" "))
            end_sents_len = len(" ".join(sentences[-self.end :]).split(" "))
            if (len(text.split(" ")) - end_sents_len - start_sents_len) < self.k * self.span_length:
                return None
            while left_to_remove:
                index_to_remove = random.sample(range(start_sents_len, len(text_copy.split(" ")) - end_sents_len), k=1)[
                    0
                ]
                augmented_words = [
                    word
                    for i, word in enumerate(text_copy.split(" "))
                    if i not in range(index_to_remove, index_to_remove + self.span_length)
                ]
                text_copy = " ".join(augmented_words)
                left_to_remove -= 1
            return text_copy
        else:
            raise ValueError()

    def __str__(self) -> str:
        return f"{self.type_}-removal"


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


class FasttextSwap(Augmentation):
    def __init__(
        self,
        model_path: str = "./data/augmentation_models/cc.he.300.bin",
        word_p: float = 0.05,
    ) -> None:
        assert 0.0 <= word_p <= 1.0, "the word replacement probability can only be between 0 and 1"

        self.model = fasttext.load_model(model_path)
        fasttext.util.reduce_model(self.model, 100)
        self.word_p = word_p

    def __call__(self, text: str) -> str | None:
        words: list[str] = text.split()
        for word_idx, word in enumerate(words):
            if random.random() < self.word_p:
                neighbors = self.model.get_nearest_neighbors(word=word, k=5)
                words[word_idx] = random.sample(neighbors, k=1)[0][1]

        return " ".join(words)

    def __str__(self) -> str:
        return "fasttext"


class ChainedAugmentation(Augmentation):
    def __init__(self, augmentations: list["Augmentation"]) -> None:
        self.augmentations = augmentations

    def __call__(self, text: str) -> str | None:
        augmented_text: str | None = text

        for augmentation in self.augmentations:
            if augmented_text is not None:
                augmented_text = augmentation(augmented_text)
                if augmented_text is None:
                    return None
            else:
                return None

        return augmented_text

    def __str__(self) -> str:
        return "+".join(str(aug) for aug in self.augmentations)


def get_augmentations(augs_type: list[str], augmentations_config: dict[str, Any] | None) -> list[Augmentation]:
    augmentations: list[Augmentation] = []
    for type_ in augs_type:
        if "+" in type_:
            types: list[str] = type_.split("+")
        else:
            types = [type_]

        sub_augmentations: list[Augmentation] = []
        for sub_type in types:
            config = augmentations_config.get(sub_type, {}) if augmentations_config is not None else {}
            match sub_type:
                case "word-removal" | "sentence-removal" | "span-removal":
                    sub_augmentations.append(BlockRemoval(**config, type_=sub_type.split("-")[0]))
                case "sentence-shuffle":
                    sub_augmentations.append(SetenceShuffle(**config))
                case "keyboard-swapping":
                    sub_augmentations.append(KeyboardSwap(**config))
                case "fasttext":
                    sub_augmentations.append(FasttextSwap(**config))
                case _:
                    raise ValueError(f"{sub_type} is not a valid augmentation")
            if len(sub_augmentations) > 1:
                augmentation: Augmentation = ChainedAugmentation(sub_augmentations)
            else:
                augmentation = sub_augmentations[0]
        augmentations.append(augmentation)
    return augmentations
