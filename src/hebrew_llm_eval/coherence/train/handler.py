# src/hebrew_llm_eval/coherence/train/handler.py

import argparse
import logging
import sys

# Import necessary enums
from ...common.enums import SplitType, TrainerType
from .core import run_training  # Import the core logic


def handle_train_cli(args: argparse.Namespace) -> None:
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

    # --- Argument Validation ---
    print("Validating arguments...")
    # 1. Splitter validation
    if args.split_type == SplitType.RANDOM:
        if args.split_key is not None:
            raise argparse.ArgumentTypeError(f"Cannot use --split-key when --split-type is '{SplitType.RANDOM.name}'.")
    else:  # split_type is not SplitType.RANDOM (e.g., KEY)
        if args.split_key is None:
            raise argparse.ArgumentTypeError(f"--split-key is required when --split-type is '{args.split_type.name}'.")

    # 2. Trainer and Loss parameter validation
    trainer_type_name = args.trainer_type.name  # For user-friendly messages

    if args.trainer_type == TrainerType.WEIGHTED:
        if args.class_weights is None:
            raise argparse.ArgumentTypeError(
                f"--class-weights <W0> <W1> ... is required when --trainer-type={trainer_type_name}"
            )
        if len(args.class_weights) != args.num_labels:
            raise argparse.ArgumentTypeError(
                f"--class-weights must have exactly {args.num_labels} values (one per class), "
                f"but received {len(args.class_weights)}: {args.class_weights}"
            )
        # Check for conflicting arguments
        if args.focal_loss_alpha is not None:
            raise argparse.ArgumentTypeError(
                f"--focal-loss-alpha cannot be used when --trainer-type={trainer_type_name}"
            )
        # Optional: Warn if gamma is set to non-default, as it will be ignored
        if args.focal_loss_gamma != 2.0:  # Check against the default value set in args.py
            logging.warning(
                f"--focal-loss-gamma was set to {args.focal_loss_gamma}, but it will be ignored "
                f"when --trainer-type={trainer_type_name}"
            )

    elif args.trainer_type == TrainerType.FOCAL:
        # Check for conflicting arguments
        if args.class_weights is not None:
            raise argparse.ArgumentTypeError(f"--class-weights cannot be used when --trainer-type={trainer_type_name}")
        # Check alpha length if provided
        if args.focal_loss_alpha is not None and len(args.focal_loss_alpha) != args.num_labels:
            raise argparse.ArgumentTypeError(
                f"--focal-loss-alpha must have exactly {args.num_labels} values (one per class) if provided, "
                f"but received {len(args.focal_loss_alpha)}: {args.focal_loss_alpha}"
            )
        # Gamma always has a value due to its default in args

    elif args.trainer_type == TrainerType.DEFAULT:
        # Check for conflicting arguments
        if args.class_weights is not None:
            raise argparse.ArgumentTypeError(f"--class-weights cannot be used when --trainer-type={trainer_type_name}")
        if args.focal_loss_alpha is not None:
            raise argparse.ArgumentTypeError(
                f"--focal-loss-alpha cannot be used when --trainer-type={trainer_type_name}"
            )
        # Optional: Warn if gamma is set to non-default
        if args.focal_loss_gamma != 2.0:
            logging.warning(
                f"--focal-loss-gamma was set to {args.focal_loss_gamma}, but it will be ignored "
                f"when --trainer-type={trainer_type_name}"
            )
    else:
        # This case should ideally not be reached if choices in argparse are set correctly
        raise ValueError(f"Unhandled trainer type during validation: {args.trainer_type}")

    print("Argument validation passed.")
    # --- End Argument Validation ---

    try:
        # Call the core training function, passing ALL relevant args
        print("Calling core training logic...")
        run_training(
            # Data/Split Args
            data_path=args.data_path,
            split_type=args.split_type,
            split_key=args.split_key,
            test_size=args.test_size,
            val_size=args.val_size,
            cv=args.cv,
            # Model/Tokenization Args
            model_name=args.model_name,
            max_length=args.max_length,
            k_max=args.k_max,
            # Training Hyperparameters
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            # Trainer/Loss Args <<-- NEW
            trainer_type=args.trainer_type,
            class_weights=args.class_weights,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            num_labels=args.num_labels,
            # Execution/Output Args
            output_dir=args.output_dir,
            device=args.device,
            no_test=args.no_test,
        )

        print("\n--- Training process completed. ---")

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
