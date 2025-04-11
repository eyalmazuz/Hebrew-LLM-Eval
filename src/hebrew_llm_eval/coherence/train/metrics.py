# src/hebrew_llm_eval/coherence/train/metrics.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction):
    """
    Computes evaluation metrics for sequence classification.

    Args:
        eval_pred: An EvalPrediction object containing predictions and label_ids.

    Returns:
        A dictionary containing f1, roc_auc, accuracy, and pr_auc scores.
    """
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    # Apply softmax to logits to get probabilities
    probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=1)
    # Get predicted class labels
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_pred.label_ids

    # Calculate metrics
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    # Use probabilities for AUC metrics
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, probs[:, 1])

    metrics = {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy, "pr_auc": pr_auc}

    return metrics
