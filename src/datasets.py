from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PairDataset(Dataset):
    def __init__(
        self,
        data: list[tuple[tuple[str, str], int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.texts = [d[0] for d in data]  # List of pairs (sentence A, sentence B)
        self.labels = [d[1] for d in data]
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length != -1 else 8192

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sent_a, sent_b = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(sent_a, sent_b, padding=True, truncation=True, max_length=self.max_length)
        encoding["label"] = label

        return encoding


class SentenceOrderingDataset(Dataset):
    def __init__(
        self,
        data: list[tuple[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.texts = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length != -1 else 8192

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length)
        encoding["label"] = label

        return encoding
