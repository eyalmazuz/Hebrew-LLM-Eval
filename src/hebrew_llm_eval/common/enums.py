# src/hebrew_llm_eval/common/enums.py
import enum


# Change from enum.Enum to enum.StrEnum
class SplitType(enum.StrEnum):
    """Enumeration for data splitting methods (using StrEnum)."""

    RANDOM = "random"
    KEY = "key"


class TrainerType(enum.StrEnum):
    """Enumeration for trainer types (using StrEnum)."""

    DEFAULT = "default"
    WEIGHTED = "weighted"
    FOCAL = "focal"
