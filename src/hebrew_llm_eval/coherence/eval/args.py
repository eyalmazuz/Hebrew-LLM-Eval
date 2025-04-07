import argparse

# Import the specific handler for the 'eval' command under 'coherence'
# Adjust path/name if you changed 'handler.py' or the function name
from .handler import handle_eval_cli  # *** Assumes handle_eval_cli exists ***


def add_eval_subcommand(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    """Adds the 'eval' subcommand to the 'coherence' parser."""
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained coherence model.",
        parents=[common_parser],  # Inherit common options like -v, -c
        description="Evaluate a trained sequence coherence model on new data.",  # More specific description
    )
    # Add arguments specific to the 'eval' action
    eval_parser.add_argument("--eval-data-path", type=str, required=True, help="Path to the evaluation data file")
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained coherence model directory"
    )
    eval_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    eval_parser.add_argument(
        "--output-file", type=str, default="./eval_results.json", help="File to save evaluation results"
    )
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (e.g., 'cuda', 'cpu')")

    # Set the default function for the 'eval' command
    eval_parser.set_defaults(func=handle_eval_cli)
