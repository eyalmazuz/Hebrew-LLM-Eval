import json
import random
from typing import Any

IDX2SOURCE = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}


def load_data(path: str) -> list[dict[str, Any]]:
    try:
        with open(path) as fd:
            summaries = [json.loads(line) for line in fd]
    except FileNotFoundError:
        raise FileNotFoundError(f"No file was found at {path}")
    except json.JSONDecodeError as e:
        # Re-raise the JSONDecodeError with the same or modified details
        raise json.JSONDecodeError(f"Error decoding JSON in file {path}: {e.msg}", e.doc, e.pos)

    return summaries


def get_train_test_split(
    summaries: list[dict[str, Any]],
    split_type: str,
    source_type: str,
    test_size: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_type.lower() == "random":
        if test_size is not None:
            random.shuffle(summaries)
            train_set = summaries[int(len(summaries) * test_size) :]
            test_set = summaries[: int(len(summaries) * test_size)]
        else:
            raise ValueError("Test size can't be None")

    elif split_type.lower() == "source":
        train_set = [summary for summary in summaries if summary["metadata"]["source"] != source_type]

        test_set = [summary for summary in summaries if summary["metadata"]["source"] == source_type]

    else:
        raise ValueError(f"Invlid split type was selected {split_type}")

    return train_set, test_set


def extract_texts(
    summaries: list[dict[str, Any]],
    only_summaries: bool = True,
    use_ai_summaries: bool = True,
) -> list[str]:
    positives: list[str] = []
    for summary in summaries:
        if (
            not only_summaries
            and "text_raw" in summary
            and summary["text_raw"] is not None
            and summary["text_raw"] != ""
        ):
            positives.append(summary["text_raw"])

        if (
            use_ai_summaries
            and "ai_summary" in summary["metadata"]
            and summary["metadata"]["ai_summary"] is not None
            and summary["metadata"]["ai_summary"] != ""
        ):
            positives.append(summary["metadata"]["ai_summary"])

        if "summary" in summary and summary["summary"] is not None and summary["summary"] != "":
            positives.append(summary["summary"])

    return positives
