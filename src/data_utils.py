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
        sentences = text.strip().split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences into pairs
        pairs += [((sentences[i], sentences[i + 1]), 1) for i in range(0, len(sentences) - 1, 2)]

    return pairs


def generate_negative_pairs(
    positive_texts: list[str],
    negative_count: int,
    negative_type: str,
) -> list[tuple[tuple[str, str], int]]:
    negatives: list[tuple[tuple[str, str], int]] = []

    for i, text in enumerate(positive_texts):
        sentences = text.split(". ")
        for j, sentence in enumerate(sentences):
            match negative_type.lower():
                case "in-pair":
                    potential_negatives = [sent for sent in sentence if sent != sentence]
                case "any":
                    numbers = [k for k in range(1, len(positive_texts) + 1) if k != i]
                    negative_texts = [positive_texts[k].split(". ") for k in random.sample(numbers, negative_count)]
                    potential_negatives = [item for sublist in negative_texts for item in sublist]
            k_negs = random.sample(potential_negatives, k=min(negative_count, len(potential_negatives)))
            negatives.extend([((sentence, negative), 0) for negative in k_negs])

    return negatives
