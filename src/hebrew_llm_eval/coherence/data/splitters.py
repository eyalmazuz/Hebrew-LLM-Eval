import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence

from ...common.enums import SplitType
from ..data.utils import get_data_split
from .types import DataRecord


class BaseSplitter(ABC):
    def __init__(self, data: MutableSequence[DataRecord]) -> None:
        self.data = data

    @abstractmethod
    def get_splits(self) -> Iterable[tuple[Iterable[DataRecord], ...]]:
        raise NotImplementedError


class RandomSplitter(BaseSplitter):
    def __init__(
        self,
        data: MutableSequence[DataRecord],
        test_size: float = 0.2,
        val_size: float = 0.2,
        num_splits: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(data)
        self.test_size = test_size
        effective_val_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
        self.val_size = effective_val_size
        self.num_splits = num_splits

    def get_splits(self) -> Iterable[tuple[Iterable[DataRecord], ...]]:
        for _ in range(self.num_splits):
            train_set, test_set = get_data_split(self.data, self.test_size)
            train_set, val_set = get_data_split(train_set, self.val_size)

            yield train_set, val_set, test_set

    def __str__(self):
        return "RandomSplitter"


class GroupSplitter(BaseSplitter):
    def __init__(self, data: list[DataRecord], split_key: str | None, **kwargs) -> None:
        super().__init__(data)
        assert split_key is not None, "key must be provided for group split"
        self.split_key = split_key
        self.groups = self._get_groups()

    def _get_groups(self) -> list[str]:
        groups: list[str] = []
        for record in self.data:
            group_key = getattr(record, self.split_key)
            if group_key not in groups:
                groups.append(group_key)
        return groups

    def get_splits(self) -> Iterable[tuple[Iterable[DataRecord], ...]]:
        group_keys = list(self.groups)
        for test_group in group_keys:
            val_group = random.choice([g for g in group_keys if g != test_group])
            train_groups = [g for g in group_keys if g not in [test_group, val_group]]
            print(f"Test group: {test_group} | Val group: {val_group} | Train groups: {train_groups}")

            test_data = [record for record in self.data if getattr(record, self.split_key) == test_group]
            val_data = [record for record in self.data if getattr(record, self.split_key) == val_group]
            train_data = [record for record in self.data if getattr(record, self.split_key) == train_groups]

            yield train_data, val_data, test_data

    def __str__(self):
        return f"GroupSplitter: {self.split_key}"


def get_split_by_type(data: list[DataRecord], split_type: SplitType, **kwargs) -> BaseSplitter:
    match split_type:
        case SplitType.RANDOM:
            return RandomSplitter(data, **kwargs)
        case SplitType.KEY:
            return GroupSplitter(data, **kwargs)
        case _:
            raise ValueError(f"Unknown split type: {split_type}")
