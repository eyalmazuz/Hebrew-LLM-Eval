# src/hebrew_llm_eval/coherence/train/args.py

import argparse

import torch  # Needed for potential device check later if desired

# Assuming TrainerType is defined in common/enums.py
from ...common.enums import SplitType, TrainerType
from .handler import handle_train_cli


def add_train_subcommand(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    """Adds the 'train' subcommand to the 'coherence' parser."""
    train_parser = subparsers.add_parser(
        "train",
        help="Train a coherence model.",
        parents=[common_parser],  # Inherit common options like -v, -c
        description="Train a model to evaluate sequence coherence.",  # More specific description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows defaults in help
    )

    # --- Data and Splitting Arguments ---
    data_group = train_parser.add_argument_group("Data and Splitting Configuration")
    data_group.add_argument(
        "--data-path", type=str, required=True, help="Path to the training data file (JSONL format)"
    )
    data_group.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for the test set")
    data_group.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the remaining data (after test split) for validation set",
    )
    data_group.add_argument(
        "--split-type",
        type=SplitType,
        default=SplitType.RANDOM,
        choices=list(SplitType),
        help="Method for splitting data. Requires --split-key if not RANDOM.",
    )
    data_group.add_argument(
        "--split-key",
        type=str,
        default=None,
        help="The key (e.g., 'source') used for group splitting (required if --split-type=KEY).",
    )
    data_group.add_argument(
        "--cv",
        type=int,
        default=1,
        help="Number of cross-validation folds (uses selected split-type within each fold if > 1)",
    )

    # --- Model and Tokenizer Arguments ---
    model_group = train_parser.add_argument_group("Model and Tokenizer Configuration")
    model_group.add_argument(
        "--model_name", type=str, default="dicta-il/alephbertgimmel-base", help="Base model name/path from Hugging Face"
    )
    model_group.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenization")
    model_group.add_argument(
        "--k-max", type=int, default=20, help="Maximum number of unique shuffles (negative examples) per original text"
    )

    # --- Training Hyperparameters ---
    hp_group = train_parser.add_argument_group("Training Hyperparameters")
    hp_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    hp_group.add_argument("--batch-size", type=int, default=32, help="Batch size per device for training")
    hp_group.add_argument("--learning-rate", type=float, default=5e-5, help="Initial learning rate for AdamW optimizer")
    # Add other TrainingArguments like weight_decay, warmup_steps etc. if needed

    # --- Custom Trainer and Loss Arguments ---
    loss_group = train_parser.add_argument_group("Trainer and Loss Configuration")
    loss_group.add_argument(
        "--trainer-type",
        type=TrainerType,
        default=TrainerType.DEFAULT,
        choices=list(TrainerType),
        help="Type of Trainer to use, allows for custom loss functions.",
    )
    loss_group.add_argument(
        "--class-weights",
        type=float,
        nargs="+",  # Expects one or more floats (e.g., 1.0 20.0)
        default=None,
        help="List of weights for each class (e.g., for class 0, class 1). "
        "Required and only used if --trainer-type=WEIGHTED. "
        "Number of weights must match number of labels (usually 2).",
    )
    loss_group.add_argument(
        "--focal-loss-alpha",
        type=float,
        nargs="+",  # Expects one or more floats
        default=None,
        help="List of alpha weighting factors for each class for Focal Loss. "
        "Only used if --trainer-type=FOCAL. If None, no alpha weighting applied in Focal Loss. "
        "Number of weights must match number of labels (usually 2).",
    )
    loss_group.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=2.0,  # Common default value
        help="Gamma focusing parameter for Focal Loss. Only used if --trainer-type=FOCAL.",
    )
    loss_group.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of training labels the model have",
    )

    # --- Execution and Output Arguments ---
    exec_group = train_parser.add_argument_group("Execution and Output Configuration")
    exec_group.add_argument(
        "--output-dir",
        type=str,
        default="./results/coherence_training",
        help="Directory to save checkpoints, logs, and final model",
    )
    exec_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training ('cuda' or 'cpu')",
    )
    exec_group.add_argument(
        "--no-test", action="store_true", help="Skip the final evaluation on the test set after training"
    )
    # Add --report-to (e.g., "wandb") argument if needed

    # Set the default function for the 'train' command
    train_parser.set_defaults(func=handle_train_cli)  # handle_train_cli should validate args combinations
