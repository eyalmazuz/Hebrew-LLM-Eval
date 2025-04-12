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
    k_values_for_metrics: list[int] = [3, 5, 10],  # K values for Recall and NDCG
) -> dict[str, float]:
    """
    Evaluates a ranking model using Top-1 accuracy, Pairwise accuracy, MRR,
    Recall@K, NDCG@K, and analyzes the impact of the number of negative samples.

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
            num_labels=2,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            attn_implementation="sdpa",  # Use "eager" if sdpa gives errors
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # bfloat16 only on CUDA
        ).to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- Metric Accumulators ---
    top_correct_count = 0
    pair_correct_count = 0
    pair_total_count = 0
    total_mrr_sum = 0.0
    evaluated_items_count = 0  # Denominator for most metrics

    # For len_shuffled analysis
    total_len_shuffled_sum = 0
    len_shuffled_success_list = []
    len_shuffled_failure_list = []

    # For Recall@K and NDCG@K
    k_values_for_metrics.sort()  # Ensure K values are sorted if needed later
    recall_k_correct_counts = {k: 0 for k in k_values_for_metrics}
    ndcg_k_sum = {k: 0.0 for k in k_values_for_metrics}
    # --------------------------

    print(f"Evaluating on {len(test_dataset)} items...")
    for i in trange(len(test_dataset), desc="Ranking Evaluation"):
        encodings, len_shuffled = test_dataset[i]

        # Skip items that couldn't be shuffled (no negatives to rank against)
        if len_shuffled == 0:
            # print(f"  --> Skipping item {i} for ranking (cannot shuffle)")
            continue  # Skip to the next item

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            # Assuming the positive class is at index 1
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            # Sort scores to find rank (higher score is better)
            # Note: probs[0] is the score for the original correct document
            sorted_scores, sorted_indices = torch.sort(probs, descending=True, stable=True)

            # Find the 1-based rank of the original document (index 0)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

            # --- Update Metrics ---
            evaluated_items_count += 1

            # Top-1 Accuracy (Recall@1)
            is_top1_correct = rank == 1  # More direct than comparing max probs
            if is_top1_correct:
                top_correct_count += 1

            # Pairwise Accuracy
            pair_correct_count += int(torch.sum(probs[0] > probs[1:]))
            pair_total_count += len_shuffled  # Number of pairs compared in this item

            # MRR
            total_mrr_sum += 1.0 / rank

            # Recall@K
            for k in k_values_for_metrics:
                if rank <= k:
                    recall_k_correct_counts[k] += 1

            # NDCG@K
            for k in k_values_for_metrics:
                if rank <= k:
                    # IDCG is 1 for binary relevance when the item is in top K
                    ndcg_k_sum[k] += 1.0 / math.log2(rank + 1)

            # len_shuffled Analysis
            total_len_shuffled_sum += len_shuffled
            if is_top1_correct:
                len_shuffled_success_list.append(len_shuffled)
            else:
                len_shuffled_failure_list.append(len_shuffled)
            # ----------------------

    # --- Calculate Final Metrics ---
    results: dict[str, float] = {}

    if evaluated_items_count > 0:
        results["evaluated_items_count"] = evaluated_items_count
        results["top_1_accuracy"] = top_correct_count / evaluated_items_count
        # Note: Original MRR divided by len(test_dataset). Changed to evaluated_items_count
        # for consistency with other metrics that only consider ranked items.
        results["mean_reciprocal_rank"] = total_mrr_sum / evaluated_items_count

        for k in k_values_for_metrics:
            results[f"recall@{k}"] = recall_k_correct_counts[k] / evaluated_items_count
            results[f"ndcg@{k}"] = ndcg_k_sum[k] / evaluated_items_count

        results["avg_len_shuffled"] = total_len_shuffled_sum / evaluated_items_count
        # Use np.mean for safe handling of empty lists (returns nan, which we convert to 0)
        results["avg_len_shuffled_success"] = (
            float(np.mean(len_shuffled_success_list)) if len_shuffled_success_list else 0.0
        )
        results["avg_len_shuffled_failure"] = (
            float(np.mean(len_shuffled_failure_list)) if len_shuffled_failure_list else 0.0
        )
        results["top_1_success_count"] = len(len_shuffled_success_list)
        results["top_1_failure_count"] = len(len_shuffled_failure_list)

    else:
        print("Warning: No items were evaluated for ranking.")
        # Fill results with zeros or NaNs if no items were evaluated
        results["evaluated_items_count"] = 0
        metrics_keys = [
            "top_1_accuracy",
            "mean_reciprocal_rank",
            "avg_len_shuffled",
            "avg_len_shuffled_success",
            "avg_len_shuffled_failure",
            "top_1_success_count",
            "top_1_failure_count",
        ]
        for k in k_values_for_metrics:
            metrics_keys.extend([f"recall@{k}", f"ndcg@{k}"])
        for key in metrics_keys:
            results[key] = 0.0

    # Pairwise accuracy has a different denominator
    if pair_total_count > 0:
        results["pair_ranking_accuracy"] = pair_correct_count / pair_total_count
    else:
        results["pair_ranking_accuracy"] = 0.0
        results["pair_total_count"] = 0
        results["pair_correct_count"] = 0

    results["pair_total_count"] = pair_total_count
    results["pair_correct_count"] = pair_correct_count

    # --- Print Summary ---
    print("\n--- Evaluation Summary ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("------------------------\n")

    return results
