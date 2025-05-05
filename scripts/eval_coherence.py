import argparse
import json
import math
import random
from itertools import permutations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the evaluation data file")
    parser.add_argument("--model-name", type=str, required=True, help="Path to the trained coherence model directory")
    parser.add_argument("--output-path", type=str, default="./results/", help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (e.g., 'cuda', 'cpu')")

    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of training labels the model have",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument(
        "--k-max", type=int, default=20, help="Maximum number of unique shuffles (negative examples) per original text"
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None


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


class ShuffleRankingDataset(Dataset):
    def __init__(self, texts: list[str], k_max: int, tokenizer, max_length: int = -1) -> None:
        self.texts: list[str] = texts
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


def main() -> None:
    args = parse_args()

    # Using your actual load_data function
    print(f"Loading data from: {args.data_path}")
    test_data = load_data(args.data_path)
    if test_data is None:
        print("Error: No data loaded. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Using your actual ShuffleDataset

    print(f"Loading model {args.model_name} with {args.num_labels=}")
    # Use the num_labels argument passed to the function
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,  # Use the passed argument
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",  # Assuming this remains constant
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
        if "cuda" in args.device and torch.cuda.is_bf16_supported()
        else torch.float32,  # Check bf16 support
    ).to(args.device)

    test_dataset = ShuffleRankingDataset(
        test_data["text"].tolist(), args.k_max, tokenizer=tokenizer, max_length=args.max_length
    )

    k_values_for_metrics: list[int] = [3, 5, 10]

    # --- Metric Accumulators ---
    top_correct_count = 0
    pair_correct_count = 0
    pair_total_count = 0
    total_mrr_sum = 0.0
    evaluated_items_count = 0

    # For len_shuffled analysis
    total_len_shuffled_sum = 0
    len_shuffled_success_list = []
    len_shuffled_failure_list = []

    # For Recall@K and NDCG@K
    k_values_for_metrics = sorted(list(set(k_values_for_metrics)))  # Ensure unique and sorted
    recall_k_correct_counts = {k: 0 for k in k_values_for_metrics}
    ndcg_k_sum = {k: 0.0 for k in k_values_for_metrics}

    # --- Added: For Score Margin and Stats Analysis ---
    margins_victory = []
    margins_defeat = []
    positive_scores = []  # Store score of the correct item (probs[0])
    negative_scores = []  # Store scores of *all* negative items (probs[1:])
    # --------------------------------------------------

    pair_score_list: list[float] = []

    print(f"Evaluating on {len(test_dataset)} items...")
    for i in trange(len(test_dataset), desc="Ranking Evaluation"):
        encodings, len_shuffled = test_dataset[i]

        if len_shuffled == 0:
            print("Failed to predict on item with idx: {i}")
            pair_score_list.append(-1.0)
            continue

        encodings = {k: v.to(args.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            # Assuming positive class is index 1
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            # --- Basic Checks ---
            expected_len = 1 + len_shuffled
            if probs.shape[0] != expected_len:
                print(f"Warning: Probs shape {probs.shape} mismatch expected {expected_len} for item {i}. Skipping.")
                continue
            # --------------------

            # Sort scores to find rank (higher score is better)
            # Using stable=True ensures consistent ranking when scores are tied.
            sorted_scores, sorted_indices = torch.sort(probs, descending=True, stable=True)

            # Find the 1-based rank of the original document (index 0)
            rank_tensor = (sorted_indices == 0).nonzero(as_tuple=True)[0]
            if rank_tensor.numel() == 0:
                print(f"Warning: Index 0 not found for item {i}. Skipping.")
                continue
            rank = rank_tensor.item() + 1

            # --- Update Core Metrics ---
            evaluated_items_count += 1
            is_top1_correct = rank == 1
            if is_top1_correct:
                top_correct_count += 1

            pair_correct_count += int(torch.sum(probs[0] > probs[1:]))
            pair_total_count += len_shuffled
            total_mrr_sum += 1.0 / rank

            pair_score_list.append(int(torch.sum(probs[0] > probs[1:])) / len_shuffled)
            # -------------------------

            for k in k_values_for_metrics:  # Recall@K & NDCG@K
                if rank <= k:
                    recall_k_correct_counts[k] += 1
                    ndcg_k_sum[k] += 1.0 / math.log2(rank + 1)
            # -------------------------

            # --- Update Analysis Metrics ---
            # len_shuffled Analysis
            total_len_shuffled_sum += len_shuffled
            if is_top1_correct:
                len_shuffled_success_list.append(len_shuffled)
            else:
                len_shuffled_failure_list.append(len_shuffled)

            # --- Added: Score Stats & Margins ---
            positive_scores.append(probs[0].item())
            if len_shuffled > 0:
                negative_scores.extend(probs[1:].tolist())
                max_neg_prob = torch.max(probs[1:]).item()
                if is_top1_correct:
                    margins_victory.append(probs[0].item() - max_neg_prob)
                else:
                    margins_defeat.append(max_neg_prob - probs[0].item())
            # ----------------------------------
            # ---------------------------

    # --- Calculate Final Metrics ---
    results: dict[str, float | int] = {}  # Allow ints for counts

    if evaluated_items_count > 0:
        results["evaluated_items_count"] = evaluated_items_count
        results["top_1_accuracy"] = top_correct_count / evaluated_items_count
        results["mean_reciprocal_rank"] = total_mrr_sum / evaluated_items_count

        for k in k_values_for_metrics:
            results[f"recall@{k}"] = recall_k_correct_counts[k] / evaluated_items_count
            results[f"ndcg@{k}"] = ndcg_k_sum[k] / evaluated_items_count

        results["avg_len_shuffled"] = total_len_shuffled_sum / evaluated_items_count
        results["avg_len_shuffled_success"] = (
            float(np.mean(len_shuffled_success_list)) if len_shuffled_success_list else 0.0
        )
        results["avg_len_shuffled_failure"] = (
            float(np.mean(len_shuffled_failure_list)) if len_shuffled_failure_list else 0.0
        )
        results["top_1_success_count"] = len(len_shuffled_success_list)  # = top_correct_count
        results["top_1_failure_count"] = len(len_shuffled_failure_list)

        # --- Added: Score Margin and Stats Results ---
        results["avg_margin_victory"] = float(np.mean(margins_victory)) if margins_victory else 0.0
        results["avg_margin_defeat"] = float(np.mean(margins_defeat)) if margins_defeat else 0.0
        results["avg_positive_score"] = float(np.mean(positive_scores)) if positive_scores else 0.0
        results["std_positive_score"] = float(np.std(positive_scores)) if positive_scores else 0.0
        # Note: avg/std negative score is across *all* negative examples seen
        results["avg_negative_score"] = float(np.mean(negative_scores)) if negative_scores else 0.0
        results["std_negative_score"] = float(np.std(negative_scores)) if negative_scores else 0.0
        # ---------------------------------------------

    else:  # Handle case where no items could be evaluated
        print("Warning: No items were evaluated for ranking.")
        results["evaluated_items_count"] = 0
        metrics_keys = [
            "top_1_accuracy",
            "mean_reciprocal_rank",
            "avg_len_shuffled",
            "avg_len_shuffled_success",
            "avg_len_shuffled_failure",
            "top_1_success_count",
            "top_1_failure_count",
            # --- Added ---
            "avg_margin_victory",
            "avg_margin_defeat",
            "avg_positive_score",
            "std_positive_score",
            "avg_negative_score",
            "std_negative_score",
            # -------------
        ]
        for k in k_values_for_metrics:
            metrics_keys.extend([f"recall@{k}", f"ndcg@{k}"])
        for key in metrics_keys:
            results[key] = 0.0

    # Pairwise accuracy uses total pairs compared as denominator
    results["pair_total_count"] = pair_total_count
    results["pair_correct_count"] = pair_correct_count
    results["pair_ranking_accuracy"] = (pair_correct_count / pair_total_count) if pair_total_count > 0 else 0.0

    # --- Print Summary ---
    print("\n--- Evaluation Summary ---")
    # Define preferred print order
    print_order = [
        "total_items_in_dataset",
        "evaluated_items_count",
        "top_1_success_count",
        "top_1_failure_count",
        "top_1_accuracy",
        "mean_reciprocal_rank",
        "pair_ranking_accuracy",
        "pair_correct_count",
        "pair_total_count",
        "avg_len_shuffled",
        "avg_len_shuffled_success",
        "avg_len_shuffled_failure",
        # --- Added ---
        "avg_positive_score",
        "std_positive_score",
        "avg_negative_score",
        "std_negative_score",
        "avg_margin_victory",
        "avg_margin_defeat",
        # -------------
    ]
    recall_keys = sorted([f"recall@{k}" for k in k_values_for_metrics], key=lambda x: int(x.split("@")[1]))
    ndcg_keys = sorted([f"ndcg@{k}" for k in k_values_for_metrics], key=lambda x: int(x.split("@")[1]))
    print_order.extend(recall_keys)
    print_order.extend(ndcg_keys)

    # Add total items from dataset at the beginning
    results["total_items_in_dataset"] = len(test_dataset)

    for key in print_order:
        if key in results:
            value = results[key]
            precision = 6 if ("std_" in key or "margin_" in key) else 4
            if isinstance(value, float):
                print(f"{key:<30}: {value:.{precision}f}")
            else:
                print(f"{key:<30}: {value}")

    # Print any remaining keys
    printed_keys = set(print_order)
    for key, value in results.items():
        if key not in printed_keys:
            precision = 6 if ("std_" in key or "margin_" in key) else 4
            if isinstance(value, float):
                print(f"{key:<30}: {value:.{precision}f}")
            else:
                print(f"{key:<30}: {value}")

    print("------------------------\n")

    with open(f"{args.output_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    test_data["pair_score"] = pair_score_list
    test_data.to_csv(f"{args.output_path}/test_data_predicted.csv", index=False)

    print(f"Results saved to {args.output_path}/results.json")


if __name__ == "__main__":
    main()
