import argparse
import sys  # Import sys for error handling

from .coherence.args import add_coherence_subparser


def parse_args(argv=None) -> argparse.Namespace:  # Added argv=None for testability
    """Creates the parser, adds subparsers, and parses arguments."""
    if argv is None:
        argv = sys.argv[1:]  # Default to system args if none provided

    parser = argparse.ArgumentParser(
        description="Hebrew LLM Eval CLI Tool"  # Example description
    )

    common_options_parser = argparse.ArgumentParser(add_help=False)
    common_options_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    common_options_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="Path to a configuration file.",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,  # Make sure a command is always required
        help="Available modes (subcommands)",
    )

    # Add subparsers by calling the imported functions
    add_coherence_subparser(subparser=subparsers, common_parser=common_options_parser)
    args = parser.parse_args(argv)
    return args


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        # 1. Parse the arguments
        args = parse_args()

        # 2. Execute the function associated with the chosen subcommand
        # This relies on set_defaults(func=...) being called in add_*_subparser functions
        if hasattr(args, "func") and callable(args.func):
            # Call the function (e.g., handle_train_cli) stored in args.func
            # Pass the parsed arguments to it
            args.func(args)
        else:
            # This should ideally not happen if required=True and all subparsers use set_defaults
            print(f"Error: No function associated with command '{args.mode}'.", file=sys.stderr)
            # You might want to print help here: parse_args(['--help']) or similar logic
            sys.exit(1)  # Exit with an error code

    except SystemExit:
        # Allow SystemExit (raised by argparse on -h/--help or errors) to propagate
        raise
    except argparse.ArgumentError as e:
        print(f"Argument Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors during argument parsing or execution
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        # Consider adding more specific error handling based on your core functions
        # import traceback
        # traceback.print_exc() # Uncomment for detailed debugging info
        sys.exit(1)  # Exit with a generic error code


if __name__ == "__main__":
    # Allows running the cli.py script directly for development/testing
    main()
