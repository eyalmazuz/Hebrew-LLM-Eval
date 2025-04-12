from dataclasses import dataclass


@dataclass
class DataRecord:
    """Represents a single record from the JSONL file."""

    text_raw: str
    summary: str
    source: str | None = None
