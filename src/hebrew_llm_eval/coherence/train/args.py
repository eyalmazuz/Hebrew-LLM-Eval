import argparse

from ...common.enums import SplitType  # Relative import from train/args.py
from .handler.train import handle_train_cli


def add_train_subcommand(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    """Adds the 'train' subcommand to the 'coherence' parser."""
    train_parser = subparsers.add_parser(
        "train",
        help="Train a coherence model.",
        parents=[common_parser],  # Inherit common options like -v, -c
        description="Train a model to evaluate sequence coherence.",  # More specific description
    )
    # Add arguments specific to the 'train' action
    train_parser.add_argument("--data-path", type=str, required=True, help="Path to the training data file")
    train_parser.add_argument(
        "--model_name", type=str, default="dicta-il/alephbertgimmel-base", help="Base model name for training"
    )
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Size of the test set")
    train_parser.add_argument("--val-size", type=float, default=0.2, help="Size of the validation set")

    train_parser.add_argument(
        "--split-type",
        type=SplitType,  # Use the Enum class directly as the type
        default=SplitType.RANDOM,  # Default to the Enum member
        # Dynamically generate choices from Enum values for help message and validation
        choices=list(SplitType),
        help="Method for splitting data. "
        f"If not '{SplitType.RANDOM}', --split-key is required. "
        f"Default: '{SplitType.RANDOM}'.",
        # Choices are automatically listed by argparse based on 'choices' list
    )
    train_parser.add_argument(
        "--split-key",
        type=str,
        default=None,
        help="The key (e.g., column name) used for non-random splitting. "
        f"Required if --split-type is not '{SplitType.RANDOM}'.",
    )

    train_parser.add_argument("--k-max", type=int, default=20, help="Maximum number of shuffles per text")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for training")
    train_parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save training results")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device for training (e.g., 'cuda', 'cpu')")
    train_parser.add_argument("--max-length", type=int, default=512, help="the maximum length data")
    train_parser.add_argument("--no-test", action="store_true", help="If not to run test after training")
    train_parser.add_argument("--cv", type=int, default=1, help="Number of cross-validation folds")

    # Set the default function for the 'train' command
    # This function (handle_train_cli) is responsible for validating the relationship
    # between --split-type and --split-key after parsing.
    train_parser.set_defaults(func=handle_train_cli)  # Assumes handle_train_cli exists
