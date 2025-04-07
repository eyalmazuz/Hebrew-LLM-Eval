import argparse
import logging  # Can configure logging level based on verbosity
import sys

from .core import run_training  # Import the core logic

# def handle_train_cli(args):
#     texts = load_data(args.data_path)
#     train_set, test_set = get_train_test_split(texts, args.test_size)
#     train_set, val_set = get_train_test_split(train_set, args.val_size)

#     train_dataset = ShuffleDataset(train_set, args.k_max, args.model_name)
#     val_dataset = ShuffleDataset(val_set, args.k_max, args.model_name)

#     run_training()


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

    try:
        # Call the core training function, passing extracted args
        run_training(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            test_size=args.test_size,
            val_size=args.val_size,
            max_length=args.max_length,
            k_max=args.k_max,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
        )

        # Display results to the user via CLI
        print("\n--- Training Summary ---")

    except FileNotFoundError as e:
        print("\nError: Data file not found.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error code
    except Exception as e:
        print("\nAn unexpected error occurred during training:", file=sys.stderr)
        logging.error("Core training failed:", exc_info=True)  # Log the full traceback for debugging
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error code

    print("\n--- Training command finished successfully. ---")
