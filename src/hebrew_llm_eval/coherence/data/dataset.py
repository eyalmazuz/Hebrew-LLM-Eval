import torch
from torch.utils.data import Dataset  # type: ignore
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .utils import generate_unique_shuffles


class ShuffleDataset(Dataset):
    def __init__(self, texts: list[str], k_max: int, tokenizer_name, max_length: int = -1) -> None:
        shuffled_texts = []
        for text in tqdm(texts):
            shuffled_texts.extend(generate_unique_shuffles(text, k_max))

        self.texts = texts
        self.labels = [1] * len(texts)
        self.texts.extend(shuffled_texts)
        self.labels.extend([0] * len(shuffled_texts))
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.texts)

    def collate(self, batch: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
        texts, labels = zip(*batch)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        inputs["labels"] = torch.tensor(labels)
        return inputs
