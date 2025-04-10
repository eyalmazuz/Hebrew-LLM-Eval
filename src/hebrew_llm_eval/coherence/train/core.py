import os
import statistics
import uuid  # For generating unique group IDs

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

# Assume these imports are correct and point to your project's modules
from ..data.dataset import ShuffleDataset
from ..data.utils import get_train_test_split, load_data
from ..evaluation_logic import ranking_eval


# --- compute_metrics function remains the same ---
def compute_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_pred.label_ids

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, probs[:, 1])

    metrics = {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy, "pr_auc": pr_auc}

    return metrics


# --- train_and_evaluate function updated ---
# Removed wandb_run parameter; Trainer will use the active run.
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
    no_test: bool,
    device: str,
    # wandb_run parameter removed
) -> dict[str, float]:
    """
    Trains and evaluates a model on a given train/val/test split.
    Logs metrics to the currently active W&B run initialized outside this function.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using your actual ShuffleDataset
    train_dataset = ShuffleDataset(train_set, k_max, tokenizer=tokenizer, max_length=max_length)
    val_dataset = ShuffleDataset(val_set, k_max, tokenizer=tokenizer, max_length=max_length)

    print(f"Loading model {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
        attn_implementation="sdpa",  # Restored original setting
        torch_dtype=torch.bfloat16,  # Restored original setting
    ).to(device)  # Ensure model is on the correct device

    # Determine if a W&B run is active to configure reporting
    report_to = "wandb" if wandb.run else "none"  # type: ignore

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
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        optim="adamw_torch_fused",  # Restored original setting
        dataloader_pin_memory=True,
        bf16=True,  # Restored original setting
        tf32=True,  # Restored original setting
        report_to=report_to,  # Report to W&B if a run is active
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=train_dataset.collate,  # Use your actual collate function
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()
    test_results = {}
    if not no_test:
        print("Finished training, starting evaluation")
        # Using your actual ranking_eval function
        test_results = ranking_eval(
            test_data=test_set,
            model=trainer.model,  # Use the best model loaded by the trainer
            model_name_or_path=trainer.state.best_model_checkpoint
            if trainer.state.best_model_checkpoint
            else model_name,
            tokenizer=tokenizer,
            k_max=k_max,
            max_length=max_length,
            device=device,
        )

        # Log test results specific to this fold to the current W&B run
        if wandb.run and test_results:  # type: ignore
            # Prefix keys with 'test_' for clarity in W&B dashboard
            wandb.log({f"test_{k}": v for k, v in test_results.items()})  # type: ignore

    return test_results


# --- run_training function updated ---
def run_training(
    data_path: str,
    output_dir: str,
    model_name: str = "dicta-il/alephbertgimmel-base",  # Original default
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_length: int = 512,
    k_max: int = 20,
    batch_size: int = 32,
    epochs: int = 3,
    cv: int = 1,  # Original default
    no_test: bool = False,
    learning_rate: float = 5e-5,
    device: str = "cuda",  # Original default
) -> None:
    # Generate a unique group name/ID for this specific cross-validation execution
    cv_group_name = f"CV_Group_{uuid.uuid4().hex[:8]}"
    print(f"Starting {cv}-fold cross-validation with W&B Group: {cv_group_name}")

    # Using your actual load_data function
    texts = load_data(data_path)

    all_test_results: list[dict[str, float]] = []

    print(f"Starting {cv}-fold cross-validation")
    shuffle_split = ShuffleSplit(n_splits=cv, test_size=test_size, random_state=42)
    for fold, (train_index, test_index) in enumerate(shuffle_split.split(texts)):
        # --- Initialize W&B Run *Inside* the Loop for Each Fold ---
        run_name = f"Fold_{fold + 1}"
        fold_run = wandb.init(  # type: ignore
            project=os.environ.get("WANDB_PROJECT", "default_project"),  # Provide a default project name
            entity=os.environ.get("WANDB_ENTITY", None),  # Your W&B entity
            group=cv_group_name,  # Group runs from this CV execution together
            name=run_name,  # Name for this specific fold's run
            job_type="train_fold",  # Categorize the run type
            reinit=True,  # Allow re-initialization in the same script
            config={  # Log fold-specific and overall hyperparameters
                "fold": fold + 1,
                "cv_folds": cv,
                "model_name": model_name,
                "max_length": max_length,
                "k_max": k_max,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "test_size": test_size,
                "val_size": val_size,
                "data_path": data_path,
            },
        )
        # -----------------------------------------------------------

        print(f"--- Starting Fold {fold + 1}/{cv} (W&B Run ID: {fold_run.id}) ---")  # type: ignore
        train_set = [texts[i] for i in train_index]
        test_set = [texts[i] for i in test_index]

        # Using your actual get_train_test_split function
        # Calculate the proportion of the original *training* data to use for validation
        effective_val_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
        train_set, val_set = get_train_test_split(train_set, effective_val_size)

        # Update output directory for each fold
        # Using fold number is generally clear for CV outputs
        fold_output_dir = os.path.join(output_dir, f"{cv_group_name}", f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Call train_and_evaluate. It will detect and use the active 'fold_run'.
        # No need to pass the wandb_run object explicitly.
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
            no_test=no_test,
            device=device,
        )

        if not no_test and test_results:
            all_test_results.append(test_results)
            print(f"Fold {fold + 1} Test Results: {test_results}")
        else:
            print(f"Fold {fold + 1} completed (no test evaluation).")

        # --- Finish the W&B run for the current fold ---
        fold_run.finish()  # type: ignore
        # -----------------------------------------------

    # --- Post-Cross-Validation Summary ---
    print("\n--- Cross-Validation Finished ---")

    if all_test_results:
        # Calculate and print average and std for CV using original keys
        print("\n--- Aggregated Cross-Validation Results ---")
        summary_results = {}
        summary_stats_text = []

        # Calculate stats for metrics present in the results
        metric_keys = list(all_test_results[0].keys())  # Assumes keys are consistent
        for key in metric_keys:
            values: list[float] = [res[key] for res in all_test_results if res.get(key) is not None]
            if len(values) >= 1:
                avg_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                summary_results[f"{key}_avg"] = avg_val
                summary_results[f"{key}_std"] = std_val
                summary_stats_text.append(f"Avg {key}: {avg_val:.3f} (±{std_val:.3f})")
            else:
                print(f"Warning: Metric '{key}' not consistently found in fold results.")

        for line in summary_stats_text:
            print(line)

        # --- Log Summary Statistics to a *New*, Separate W&B Run ---
        print(f"\nLogging summary statistics to W&B (Group: {cv_group_name})...")
        summary_run = wandb.init(  # type: ignore
            project=os.environ.get("WANDB_PROJECT", "default_project"),  # Same project
            entity=os.environ.get("WANDB_ENTITY", None),
            group=cv_group_name,  # Use the same group name to link it
            name=f"{cv_group_name}_Summary",  # Distinct name for summary run
            job_type="cv_summary",  # Distinct job type
            reinit=True,
            config={  # Log config relevant to the overall CV process
                "cv_folds": cv,
                "model_name": model_name,
                "max_length": max_length,
                "k_max": k_max,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "test_size": test_size,
                "val_size": val_size,
                "data_path": data_path,
            },
        )

        # Log aggregated metrics to the summary section of the summary run
        summary_run.summary.update(summary_results)  # Use summary for final values

        summary_run.finish()
        print(f"Summary results logged to W&B run: {summary_run.name}")
        # ----------------------------------------------------------
    else:
        print("\nNo test results were generated across folds. Skipping summary logging.")
