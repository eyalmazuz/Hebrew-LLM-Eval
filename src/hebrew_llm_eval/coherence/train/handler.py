import argparse  # Make sure argparse is imported
import logging
import sys

from ...common.enums import SplitType
from .core import run_training  # Import the core logic


def handle_train_cli(args: argparse.Namespace) -> None:
    """
    CLI handler for the 'train' command. Parses args, calls core logic,
    and handles CLI I/O.
    """
    # Configure logging level based on verbosity argument
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )  # Use force=True if basicConfig might have been called elsewhere

    print("--- Starting Coherence Model Training (CLI Mode) ---")  # User-facing start message
    if args.verbose:
        print(f"Raw CLI arguments: {vars(args)}")  # Print args if verbose

    if args.split_type == SplitType.RANDOM:  # Compare with Enum member
        if args.split_key is not None:
            raise argparse.ArgumentTypeError(f"Cannot use --split-key when --split-type is '{SplitType.RANDOM}'.")
    else:  # split_type is not SplitType.RANDOM
        if args.split_key is None:
            # Use .value for user-friendly error message
            raise argparse.ArgumentTypeError(f"--split-key is required when --split-type is '{args.split_type}'.")
    try:
        # Call the core training function, passing extracted args
        run_training(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            test_size=args.test_size,
            val_size=args.val_size,
            split_type=args.split_type,
            split_key=args.split_key,
            max_length=args.max_length,
            k_max=args.k_max,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            cv=args.cv,
            no_test=args.no_test,
        )

        # Display results to the user via CLI
        print("\n--- Training Summary ---")

    except FileNotFoundError as e:
        print("\nError: Data file not found.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error code
    except Exception as e:
        # Note: ArgumentTypeError raised above will be caught here if not handled earlier
        print("\nAn unexpected error occurred during training:", file=sys.stderr)
        logging.error("Core training failed:", exc_info=True)  # Log the full traceback for debugging
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error code

    print("\n--- Training command finished successfully. ---")
