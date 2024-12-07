import math
import random


def k_block_shuffling(texts: list[str], permutation_count: int, block_size: int) -> list[tuple[str, int]]:
    negatives: list[tuple[str, int]] = []
    for text in texts:
        # Split the text into sentences based on periods
        sentences = text.strip().split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences into blocks of size block_size
        blocks = [". ".join(sentences[i : i + block_size]) for i in range(0, len(sentences), block_size)]

        original_order = tuple(blocks)
        num_blocks = len(blocks)

        # Compute total number of possible permutations
        total_permutations = math.factorial(num_blocks)
        max_unique_permutations = total_permutations - 1  # Exclude the original order

        # Adjust permutation_count if necessary
        negative_count = min(permutation_count, max_unique_permutations)

        # Generate random permutations
        permutations_set: set[tuple[str, ...]] = set()
        while len(permutations_set) < negative_count:
            perm = blocks[:]
            random.shuffle(perm)
            perm_tuple = tuple(perm)
            if perm_tuple != original_order:
                permutations_set.add(perm_tuple)

        negatives.extend([(". ".join(perm) + ".", 0) for perm in permutations_set])
    return negatives


def generate_nsp_pairs(
    texts: list[str],
) -> list[tuple[tuple[str, str], int]]:
    pairs: list[tuple[tuple[str, str], int]] = []
    for text in texts:
        # Split the text into sentences based on periods
        sentences = text.strip().split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences into pairs
        pairs += [((sentences[i], sentences[i + 1]), 0) for i in range(0, len(sentences) - 1)]

    return pairs


def generate_negative_pairs(
    positive_texts: list[str],
    negative_count: int,
    negative_type: str,
) -> list[tuple[tuple[str, str], int]]:
    negatives: list[tuple[tuple[str, str], int]] = []

    for i, text in enumerate(positive_texts):
        # Split into sentences and remove empty entries
        sentences = [line.strip() for line in text.split(".") if line.strip()]

        # If not enough sentences to form a positive pair, skip
        if len(sentences) < 2:
            continue

        for j, sentence in enumerate(sentences[:-1]):
            match negative_type.lower():
                case "in-pair":
                    # For in-pair negatives, select from the same text excluding current and next sentence
                    potential_negatives = [sent for k, sent in enumerate(sentences) if k != j and k != j + 1]
                case "any":
                    # For "any" negatives, pick sentences from other texts
                    candidates = [k for k in range(len(positive_texts)) if k != i]
                    # Ensure we don't sample more texts than we have
                    sample_count = min(len(candidates), negative_count)
                    chosen_indices = random.sample(candidates, k=sample_count)
                    # Collect sentences from chosen texts
                    potential_negatives = []
                    for idx in chosen_indices:
                        other_sentences = [s.strip() for s in positive_texts[idx].split(".") if s.strip()]
                        potential_negatives.extend(other_sentences)
                case _:
                    # If negative_type is unrecognized, produce no negatives
                    potential_negatives = []

            # Sample negatives if available
            if potential_negatives:
                k_negs = random.sample(potential_negatives, k=min(negative_count, len(potential_negatives)))
                negatives.extend([((sentence, neg), 1) for neg in k_negs])

    return negatives
