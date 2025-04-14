import argparse
import logging
import sys

from .core import run_evaluation


def handle_eval_cli(args: argparse.Namespace) -> None:
    """
    CLI handler for the 'train' command. Parses args, validates them,
    calls core logic, and handles CLI I/O.
    """
    # Configure logging level based on verbosity argument
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

    print("--- Starting Coherence Model Training (CLI Mode) ---")
    if args.verbose:
        # Use vars() to get a dict representation of the namespace for easy printing
        print(f"Raw CLI arguments: {vars(args)}")

    print("Argument validation passed.")
    # --- End Argument Validation ---

    try:
        # Call the core training function, passing ALL relevant args
        print("Calling core training logic...")
        run_evaluation(
            # Data/Split Args
            data_path=args.data_path,
            # Model/Tokenization Args
            model_name=args.model_name,
            max_length=args.max_length,
            k_max=args.k_max,
            # Training Hyperparameters
            batch_size=args.batch_size,
            num_labels=args.num_labels,
            output_dir=args.output_dir,
            device=args.device,
        )

        print("\n--- Eval process completed. ---")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found at '{args.data_path}'.", file=sys.stderr)
        logging.error(f"Details: {e}", exc_info=True)
        sys.exit(1)
    except argparse.ArgumentTypeError as e:  # Catch validation errors specifically
        print("\nError: Invalid argument combination.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        # Consider printing usage or help here if desired
        # parser.print_help() # Need access to the parser object for this
        sys.exit(1)
    except Exception as e:
        print("\nAn unexpected error occurred during training:", file=sys.stderr)
        # Log the full traceback for debugging
        logging.error("Core training function failed:", exc_info=True)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Training command finished successfully. ---")
