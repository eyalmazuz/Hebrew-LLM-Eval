from collections.abc import Iterable

import torch
from torch.utils.data import Dataset  # type: ignore
from tqdm.auto import tqdm

from .types import DataRecord
from .utils import generate_unique_shuffles


class ShuffleDataset(Dataset):
    def __init__(self, texts: Iterable[DataRecord], k_max: int, tokenizer, max_length: int = -1) -> None:
        shuffled_texts = []
        for text in tqdm(texts):
            shuffled_texts.extend(generate_unique_shuffles(text.summary, k_max))

        self.texts: list[str] = [text.summary for text in texts]
        self.labels = [1] * len(self.texts)
        self.texts.extend(shuffled_texts)
        self.labels.extend([0] * len(shuffled_texts))
        self.max_length = max_length

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.texts)

    def collate(self, batch: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
        texts, labels = zip(*batch)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        inputs["labels"] = torch.tensor(labels)
        return inputs


class ShuffleRankingDataset(Dataset):
    def __init__(self, texts: Iterable[DataRecord], k_max: int, tokenizer, max_length: int = -1) -> None:
        self.texts: list[str] = [text.summary for text in texts]
        self.max_length = max_length

        self.tokenizer = tokenizer
        self.k_max = k_max

    def __len__(self) -> int:
        """Returns the number of original documents."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        original_text = self.texts[idx]

        # Generate shuffled texts using the helper function
        shuffled_texts = generate_unique_shuffles(original_text, self.k_max)

        encodings = self.tokenizer(
            [original_text] + shuffled_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        return encodings, len(shuffled_texts)
