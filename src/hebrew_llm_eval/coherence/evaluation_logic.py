import math
from collections.abc import Iterable

import numpy as np  # For safe averaging
import torch
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification

from .data.dataset import ShuffleRankingDataset
from .data.types import DataRecord


def ranking_eval(
    test_data: Iterable[DataRecord],
    model,
    model_name_or_path: str,
    tokenizer,
    k_max: int,  # Max number of shuffles per item
    max_length: int,
    device: str,
    num_labels: int,
    k_values_for_metrics: list[int] = [3, 5, 10],  # K values for Recall and NDCG
) -> dict[str, float]:  # Return type hint might need adjustment if including counts
    """
    Evaluates a ranking model using Top-1 accuracy, Pairwise accuracy, MRR,
    Recall@K, NDCG@K, score margins, score statistics, and analyzes the
    impact of the number of negative samples.

    Args:
        test_data: Iterable of DataRecord objects.
        model: Pre-loaded model or None to load from path.
        model_name_or_path: Path or name for loading the model if model is None.
        tokenizer: Tokenizer corresponding to the model.
        k_max: Max number of shuffles used when creating the dataset.
        max_length: Max sequence length for tokenizer.
        device: Device to run inference on ('cuda', 'cpu').
        k_values_for_metrics: List of K values for calculating Recall@K and NDCG@K.

    Returns:
        A dictionary containing all calculated evaluation metrics.
    """
    test_dataset = ShuffleRankingDataset(test_data, k_max, tokenizer=tokenizer, max_length=max_length)

    if model is None:
        print(f"Loading model from {model_name_or_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            attn_implementation="sdpa",  # Use "eager" if sdpa gives errors
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,  # bfloat16 only on CUDA
        ).to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

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

    print(f"Evaluating on {len(test_dataset)} items...")
    for i in trange(len(test_dataset), desc="Ranking Evaluation"):
        encodings, len_shuffled = test_dataset[i]

        if len_shuffled == 0:
            continue

        encodings = {k: v.to(device) for k, v in encodings.items()}

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

    return results
