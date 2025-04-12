import json
import math
import random
from collections.abc import MutableSequence
from itertools import permutations
from typing import Any

from .types import DataRecord

# --- Dependency: NLTK ---
try:
    import nltk  # type: ignore

    # Download 'punkt' resource if not already downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("NLTK 'punkt' resource not found. Downloading...")
        nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize  # type: ignore
except ImportError:
    raise ImportError("NLTK is required for sentence splitting. Please install it: pip install nltk")
# ------------------------


IDX2SOURCE = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}


def load_data(path: str) -> list[DataRecord]:
    with open(path) as fd:
        summaries = [json.loads(line) for line in fd.readlines()]

    records = []
    for summary in summaries:
        if "summary" in summary and summary["summary"] is not None and summary["summary"] != "":
            records.append(
                DataRecord(
                    text_raw=summary["text_raw"],
                    summary=summary["summary"],
                    source=summary["metadata"].get("source", None),
                )
            )
    return records


def get_data_split(
    texts: MutableSequence[Any],
    split_size: float | None = None,
) -> tuple[MutableSequence[Any], MutableSequence[Any]]:
    if split_size is None:
        raise ValueError("Split size can't be None")
    else:
        random.shuffle(texts)
        train_set = texts[int(len(texts) * split_size) :]
        test_set = texts[: int(len(texts) * split_size)]
        return train_set, test_set


def generate_unique_shuffles(text: str, k_max: int) -> list[str]:
    """
    Generates up to k_max unique shuffled versions of the sentences in the text.
    Returns an empty list if the text has fewer than 2 sentences.
    """
    sentences = sent_tokenize(text)
    n = len(sentences)

    if n < 2:
        return []  # Cannot shuffle

    original_order_tuple = tuple(sentences)
    unique_shuffled_texts = set()

    try:
        n_available = math.factorial(n) - 1
    except (OverflowError, ValueError):  # ValueError added for potentially large n
        n_available = 10000000

    num_to_generate = min(n_available, k_max)
    if num_to_generate <= 0:  # Handle edge case if k_max is 0 or factorial calculation issue
        return []

    # Use permutations for small n if feasible and less than k_max requires it
    # Adjust threshold '9' or '10' based on practical limits
    use_permutations = False
    if n < 10:
        try:
            total_perms_count = math.factorial(n)
            if total_perms_count < 2 * k_max or total_perms_count < 1000:  # Heuristic
                use_permutations = True
        except (OverflowError, ValueError):
            pass  # Fallback to random sampling

    if use_permutations:
        all_perms = set(permutations(sentences))
        all_perms.discard(original_order_tuple)

        sampled_perms = random.sample(
            list(all_perms),
            min(len(all_perms), int(num_to_generate)),  # Ensure num_to_generate is int
        )
        for p in sampled_perms:
            unique_shuffled_texts.add(" ".join(p))

    else:
        # Fallback to random shuffling for large n or if permutations are too many
        max_attempts = int(num_to_generate * 5 + 10)  # Adjusted attempts heuristic
        attempts = 0
        while len(unique_shuffled_texts) < num_to_generate and attempts < max_attempts:
            shuffled_sentences = sentences[:]
            random.shuffle(shuffled_sentences)
            if tuple(shuffled_sentences) != original_order_tuple:
                unique_shuffled_texts.add(" ".join(shuffled_sentences))
            attempts += 1
        # Optional: Add warning if not enough unique shuffles found
        # if len(unique_shuffled_texts) < num_to_generate:
        #    print(f"Warning: Found {len(unique_shuffled_texts)}/{num_to_generate} for n={n}")

    return list(unique_shuffled_texts)


# --- End Helper Function ---
