# src/hebrew_llm_eval/coherence/train/core.py

import logging
import os
import statistics
import sys
import uuid  # For generating unique group IDs
from collections.abc import Iterable

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

import wandb

# Assume these imports are correct and point to your project's modules
from ...common.enums import SplitType, TrainerType  # Added TrainerType
from ..data.dataset import ShuffleDataset
from ..data.splitters import get_split_by_type
from ..data.types import DataRecord
from ..data.utils import load_data
from ..evaluation_logic import ranking_eval
from .metrics import compute_metrics

# Import the factory function and custom trainer classes for type hinting if needed
from .trainers import FocalLossTrainer, WeightedLossTrainer, get_trainer


# --- train_and_evaluate function updated ---
def train_and_evaluate(
    # Data sets
    train_set: Iterable[DataRecord],
    val_set: Iterable[DataRecord],
    test_set: Iterable[DataRecord],
    # Model and tokenizer args
    model_name: str,
    num_labels: int,  # Added num_labels
    max_length: int,
    # Data processing args
    k_max: int,
    # Training args
    output_dir: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    # Custom trainer args <<-- NEW
    trainer_type: TrainerType,
    class_weights_list: list[float] | None,
    focal_loss_alpha_list: list[float] | None,
    focal_loss_gamma_val: float,
    # Execution args
    no_test: bool,
    device: str,
) -> dict[str, float]:
    """
    Trains and evaluates a model on a given train/val/test split using the specified trainer type.
    Logs metrics to the currently active W&B run initialized outside this function.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using your actual ShuffleDataset
    train_dataset = ShuffleDataset(train_set, k_max, tokenizer=tokenizer, max_length=max_length)
    val_dataset = ShuffleDataset(val_set, k_max, tokenizer=tokenizer, max_length=max_length)

    print(f"Loading model {model_name} with {num_labels=}")
    # Use the num_labels argument passed to the function
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # Use the passed argument
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",  # Assuming this remains constant
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
        if "cuda" in device and torch.cuda.is_bf16_supported()
        else torch.float32,  # Check bf16 support
    ).to(device)

    # Determine if a W&B run is active to configure reporting
    report_to = "wandb" if wandb.run else "none"  # type: ignore

    current_wandb_run_name = wandb.run.name if wandb.run else None  # type: ignore
    print(f"Current W&B Run Name: {current_wandb_run_name}")

    # --- Training Arguments ---
    # These are mostly independent of the trainer type
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,  # Use same batch size for eval unless specified otherwise
        weight_decay=0.1,  # Consider making this an arg if needed
        max_grad_norm=1.0,  # Consider making this an arg if needed
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",  # Default scheduler type
        warmup_ratio=0.1,  # Example: 10% warmup, consider making an arg
        eval_strategy="steps",  # Keep eval strategy
        eval_steps=500,  # Consider making this an arg
        logging_strategy="steps",
        logging_steps=100,  # Consider making this an arg
        save_strategy="steps",
        save_steps=500,  # Consider making this an arg
        save_total_limit=1,  # Keep only the best checkpoint + the last one
        metric_for_best_model="roc_auc",  # Assuming this remains the primary metric
        greater_is_better=True,
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        bf16=bool("cuda" in device and torch.cuda.is_bf16_supported()),  # Enable bf16 only if supported
        tf32="cuda" in device,  # Enable tf32 only on CUDA
        report_to=report_to,
        run_name=current_wandb_run_name,
        load_best_model_at_end=True,  # Load the best model based on eval metric at the end
        seed=42,  # Add seed for reproducibility
    )

    # --- Select and Prepare Trainer ---
    TrainerClass = get_trainer(trainer_type)  # Get the class using the factory

    # Base arguments for all trainer classes
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "data_collator": train_dataset.collate,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "processing_class": tokenizer,  # Pass tokenizer for saving purposes
    }

    # Add specific arguments based on the trainer type
    if TrainerClass is WeightedLossTrainer:
        print("Preparing WeightedLossTrainer...")
        if class_weights_list is not None:
            # Convert list to tensor and move to device
            class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
            trainer_kwargs["class_weights"] = class_weights_tensor
            print(f"  Using class weights: {class_weights_tensor.tolist()} on device {device}")
        else:
            # This case should be caught by handler validation, but good to check
            print("  Warning: WeightedLossTrainer selected but no class weights provided. Using default loss.")

    elif TrainerClass is FocalLossTrainer:
        print("Preparing FocalLossTrainer...")
        # Always pass gamma (it has a default)
        trainer_kwargs["focal_loss_gamma"] = focal_loss_gamma_val
        print(f"  Using gamma: {focal_loss_gamma_val}")

        if focal_loss_alpha_list is not None:
            # Convert list to tensor and move to device
            focal_loss_alpha_tensor = torch.tensor(focal_loss_alpha_list, dtype=torch.float32).to(device)
            trainer_kwargs["focal_loss_alpha"] = focal_loss_alpha_tensor
            print(f"  Using alpha weights: {focal_loss_alpha_tensor.tolist()} on device {device}")
        else:
            print("  Using no alpha weighting (alpha=None)")

    else:  # Default Trainer
        print("Preparing default Hugging Face Trainer.")
        # No extra arguments needed

    # Instantiate the selected trainer class
    trainer = TrainerClass(**trainer_kwargs)

    # --- Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- Evaluation ---
    test_results = {}
    if not no_test:
        print("Starting final evaluation on test set...")
        # Use the best model loaded by the trainer
        best_model = trainer.model
        # Get the path to the best checkpoint if available, otherwise use original model name
        best_model_path = trainer.state.best_model_checkpoint if trainer.state.best_model_checkpoint else model_name
        print(f"Evaluating model from: {best_model_path}")

        test_results = ranking_eval(
            test_data=test_set,
            model=best_model,  # Pass the loaded best model
            model_name_or_path=best_model_path,  # Pass path for potential reloading if needed
            tokenizer=tokenizer,
            k_max=k_max,
            max_length=max_length,
            device=device,
            num_labels=num_labels,  # Pass num_labels to ranking_eval if needed
        )
        print(f"Test Set Evaluation Results: {test_results}")

        # Log test results specific to this fold to the current W&B run
        if wandb.run and test_results:  # type: ignore
            # Prefix keys with 'test_' for clarity in W&B dashboard
            wandb.log({f"test_{k}": v for k, v in test_results.items()})  # type: ignore
    else:
        print("Skipping final evaluation on test set.")

    # Return test results for potential aggregation
    return test_results


# --- run_training function updated ---
def run_training(
    # Data/Split Args
    data_path: str,
    split_type: SplitType = SplitType.RANDOM,
    split_key: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    cv: int = 1,
    # Model/Tokenization Args
    model_name: str = "dicta-il/alephbertgimmel-base",
    num_labels: int = 2,  # Added num_labels with default
    max_length: int = 512,
    k_max: int = 20,
    # Training Hyperparameters
    batch_size: int = 32,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    # Trainer/Loss Args <<-- NEW
    trainer_type: TrainerType = TrainerType.DEFAULT,
    class_weights: list[float] | None = None,
    focal_loss_alpha: list[float] | None = None,
    focal_loss_gamma: float = 2.0,
    # Execution/Output Args
    output_dir: str = "./results/coherence_runs",  # Example default output dir
    device: str = "cuda",
    no_test: bool = False,
) -> None:
    """
    Main function to load data, set up cross-validation (if cv > 1),
    initialize W&B, and call train_and_evaluate for each fold.
    """
    # Using your actual load_data function
    print(f"Loading data from: {data_path}")
    texts = load_data(data_path)
    if not texts:
        print("Error: No data loaded. Exiting.")
        return

    all_test_results: list[dict[str, float]] = []

    # Setup data splitter
    data_splitter = get_split_by_type(
        texts, split_type, split_key=split_key, num_splits=cv, test_size=test_size, val_size=val_size
    )

    # Generate a unique group name/ID for this specific execution (CV or single run)
    run_prefix = f"{trainer_type.name}_CV" if cv > 1 else f"{trainer_type.name}_Train"
    base_group_name = f"{run_prefix}_{os.path.basename(model_name)}_{uuid.uuid4().hex[:8]}"
    print(f"Starting {cv}-fold cross-validation with W&B Group: {base_group_name}")
    print(f"Data splitting strategy: {data_splitter}")

    # --- Cross-Validation Loop ---
    for fold, (train_set, val_set, test_set) in enumerate(data_splitter.get_splits()):
        fold_num = fold + 1
        print(f"\n--- Starting Fold {fold_num}/{cv} ---")

        # --- Initialize W&B Run *Inside* the Loop for Each Fold ---
        fold_run_name = f"{base_group_name}_Fold_{fold_num}"
        fold_output_dir = os.path.join(output_dir, base_group_name, f"fold_{fold_num}")
        os.makedirs(fold_output_dir, exist_ok=True)
        print(f"Fold output directory: {fold_output_dir}")

        # Configuration dictionary for W&B logging
        config_dict = {
            # Fold Info
            "fold": fold_num,
            "cv_folds": cv,
            # Data/Split Args
            "data_path": data_path,
            "data_splitter": str(data_splitter),  # Use string representation
            "split_type": split_type.name,
            "split_key": split_key,
            "test_size": test_size,
            "val_size": val_size,
            # Model/Tokenization Args
            "model_name": model_name,
            "num_labels": num_labels,  # Log num_labels
            "max_length": max_length,
            "k_max": k_max,
            # Training Hyperparameters
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            # Trainer/Loss Args <<-- NEW
            "trainer_type": trainer_type.name,  # Log enum name
            "class_weights": class_weights,  # Log the list or None
            "focal_loss_alpha": focal_loss_alpha,  # Log the list or None
            "focal_loss_gamma": focal_loss_gamma,  # Log the float
            # Execution args
            "device": device,
            "output_dir_fold": fold_output_dir,
        }

        try:
            # Initialize W&B for the fold
            fold_run = wandb.init(  # type: ignore
                project=os.environ.get("WANDB_PROJECT", "coherence_eval"),  # Example project name
                entity=os.environ.get("WANDB_ENTITY", None),  # Your W&B entity
                group=base_group_name,  # Group runs from this execution together
                name=fold_run_name,  # Name for this specific fold's run
                job_type="train_fold",
                reinit=True,  # Allow re-initialization in the same script/notebook
                config=config_dict,  # Log fold-specific and overall hyperparameters
            )
            print(f"W&B Run initialized for Fold {fold_num}: {fold_run.get_url() if fold_run else 'Failed'}")  # type: ignore

            # Call train_and_evaluate, passing down all necessary arguments
            fold_test_results = train_and_evaluate(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                # Model/Tokenizer
                model_name=model_name,
                num_labels=num_labels,  # Pass num_labels
                max_length=max_length,
                # Data processing
                k_max=k_max,
                # Training args
                output_dir=fold_output_dir,  # Pass fold-specific output dir
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                # Custom trainer args <<-- Pass down
                trainer_type=trainer_type,
                class_weights_list=class_weights,
                focal_loss_alpha_list=focal_loss_alpha,
                focal_loss_gamma_val=focal_loss_gamma,
                # Execution args
                no_test=no_test,
                device=device,
            )

            if not no_test and fold_test_results:
                all_test_results.append(fold_test_results)
                print(f"Fold {fold_num} Test Results: {fold_test_results}")
            else:
                print(f"Fold {fold_num} completed (no test evaluation performed or results empty).")

        except Exception as e:
            print(f"Error during Fold {fold_num}: {e}", file=sys.stderr)
            logging.error(f"Fold {fold_num} failed:", exc_info=True)
            # Optionally decide whether to continue to the next fold or exit
            # continue

        finally:
            # --- Finish the W&B run for the current fold ---
            if wandb.run:  # type: ignore
                wandb.finish()  # type: ignore
                print(f"W&B Run finished for Fold {fold_num}.")
            # -----------------------------------------------

    # --- Post-Cross-Validation Summary ---
    print("\n--- Cross-Validation Finished ---")

    if all_test_results:
        print("\n--- Aggregated Cross-Validation Test Results ---")
        summary_results = {}
        summary_stats_text = []

        # Calculate stats only if results exist
        metric_keys = list(all_test_results[0].keys())  # Assumes keys are consistent
        for key in metric_keys:
            # Collect valid float values for the metric key
            values = [res[key] for res in all_test_results if isinstance(res.get(key), int | float)]

            if len(values) >= 1:
                avg_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                summary_results[f"{key}_avg"] = avg_val
                summary_results[f"{key}_std"] = std_val
                summary_stats_text.append(f"Avg {key}: {avg_val:.4f} (±{std_val:.4f})")
            else:
                logging.warning(f"Metric '{key}' not consistently found or not numeric in fold results.")

        for line in summary_stats_text:
            print(line)

        # --- Log Summary Statistics to a *New*, Separate W&B Run ---
        print(f"\nLogging summary statistics to W&B (Group: {base_group_name})...")
        summary_config = {
            k: v for k, v in config_dict.items() if k not in ["fold", "output_dir_fold"]
        }  # Log base config
        summary_config["cv_folds"] = cv  # Ensure cv count is in summary config

        try:
            summary_run = wandb.init(  # type: ignore
                project=os.environ.get("WANDB_PROJECT", "coherence_eval"),  # Same project
                entity=os.environ.get("WANDB_ENTITY", None),
                group=base_group_name,  # Use the same group name to link it
                name=f"{base_group_name}_Summary",  # Distinct name for summary run
                job_type="cv_summary",
                reinit=True,
                config=summary_config,  # Log config relevant to the overall CV process
            )

            if summary_run:
                # Log aggregated metrics to the summary section of the summary run
                summary_run.summary.update(summary_results)  # Use summary for final values
                summary_run.finish()
                print(f"Summary results logged to W&B run: {summary_run.name}")
            else:
                print("Failed to initialize W&B run for summary.")

        except Exception as e:
            print(f"Error logging summary statistics to W&B: {e}", file=sys.stderr)
            logging.error("W&B summary logging failed:", exc_info=True)
        # ----------------------------------------------------------
    elif cv > 1:
        print("\nNo test results were generated across folds. Skipping summary logging.")
    else:
        print("\nTraining finished (single run or no test evaluation). No summary results to aggregate.")

    print("--- run_training finished ---")
