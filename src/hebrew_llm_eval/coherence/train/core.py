import os
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

import wandb

from ..data.dataset import ShuffleDataset
from ..data.utils import get_train_test_split, load_data
from ..evaluation_logic import ranking_eval


def compute_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_pred.label_ids

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, probs[:, 1])

    metrics = {"f1": f"{f1:.3f}", "roc_auc": f"{roc_auc:.3f}", "accuracy": f"{accuracy:.3f}", "pr_auc": f"{pr_auc:.3f}"}

    return metrics


def train_and_evaluate(
    train_set: list[str],
    val_set: list[str],
    test_set: list[str],
    model_name: str,
    output_dir: str,
    max_length: int,
    k_max: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    do_test: bool,
    device: str,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,  # type: ignore
) -> dict[str, float]:
    """
    Trains and evaluates a model on a given train/val/test split.

    Args:
        train_set: List of training texts.
        val_set: List of validation texts.
        test_set: List of test texts.
        model_name: Name of the pre-trained model.
        output_dir: Directory to save training results.
        max_length: Maximum sequence length.
        k_max: Maximum number of shuffles per text.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for training.
        device: Device for training (e.g., 'cuda', 'cpu').
        wandb_run: Optional wandb run object.

    Returns:
        A dictionary containing the test results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ShuffleDataset(train_set, k_max, tokenizer=tokenizer, max_length=max_length)
    val_dataset = ShuffleDataset(val_set, k_max, tokenizer=tokenizer, max_length=max_length)

    print(f"Loading model {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        metric_for_best_model="eval_roc_auc",  # Change to accuracy or any other metric
        greater_is_better=True,  # Need to change to True when using accuracy
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        bf16=True,
        tf32=True,
        report_to="wandb" if wandb_run else None,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=train_dataset.collate,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()
    test_results = {}
    if do_test:
        print("Finished training, starting evaluation")
        test_results = ranking_eval(
            test_data=test_set,
            model=model,
            model_name_or_path="",
            tokenizer=tokenizer,
            k_max=k_max,
            max_length=max_length,
            device=device,
        )
    return test_results


def run_training(
    data_path: str,
    output_dir: str,
    model_name: str = "dicta-il/alephbertgimmel-base",
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_length: int = 512,
    k_max: int = 20,
    batch_size: int = 32,
    epochs: int = 3,
    cv: int = 1,
    do_test: bool = True,
    learning_rate: float = 5e-5,
    device: str = "cuda",
) -> None:
    texts = load_data(data_path)

    wandb_run = wandb.init(  # type: ignore
        project=os.environ.get("WANDB_PROJECT", None),
        entity=os.environ.get("WANDB_ENTITY", None),
        group="Sentence_Ordering",
    )

    all_test_results: list[dict[str, float]] = []

    print(f"Starting {cv}-fold cross-validation")
    shuffle_split = ShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    for fold, (train_index, test_index) in enumerate(shuffle_split.split(texts)):
        print(f"Starting fold {fold + 1}/{cv}")
        train_set = [texts[i] for i in train_index]
        test_set = [texts[i] for i in test_index]
        train_set, val_set = get_train_test_split(train_set, val_size / (1 - test_size))

        # Update output directory for each fold
        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        test_results = train_and_evaluate(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            model_name=model_name,
            output_dir=fold_output_dir,
            max_length=max_length,
            k_max=k_max,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            do_test=do_test,
            device=device,
            wandb_run=wandb_run,
        )
        all_test_results.append(test_results)

    # Calculate and print average and std for CV
    print("\n--- Cross-Validation Results ---")
    avg_top_ranking = statistics.mean([res["top_ranking_accuracy"] for res in all_test_results])
    std_top_ranking = statistics.stdev([res["top_ranking_accuracy"] for res in all_test_results])
    avg_pair_ranking = statistics.mean([res["pair_ranking_accuracy"] for res in all_test_results])
    std_pair_ranking = statistics.stdev([res["pair_ranking_accuracy"] for res in all_test_results])

    print(f"Average Top Ranking Accuracy: {avg_top_ranking:.3f} (±{std_top_ranking:.3f})")
    print(f"Average Pair Ranking Accuracy: {avg_pair_ranking:.3f} (±{std_pair_ranking:.3f})")

    if wandb_run is not None:
        wandb_run.summary["top_ranking_accuracy_avg"] = avg_top_ranking  # type: ignore
        wandb_run.summary["pair_ranking_accuracy_std"] = std_top_ranking  # type: ignore
        wandb_run.summary["top_ranking_accuracy_avg"] = avg_pair_ranking  # type: ignore
        wandb_run.summary["pair_ranking_accuracy_std"] = std_pair_ranking  # type: ignore

    if wandb_run:
        wandb_run.finish()
