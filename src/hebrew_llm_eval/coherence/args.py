import argparse

from .eval.args import add_eval_subcommand
from .train.args import add_train_subcommand


def add_coherence_subparser(subparser: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    """
    Adds the main 'coherence' command and delegates definition of its
    subcommands ('train', 'eval') to other functions.
    """

    # 1. Create the parser for the main 'coherence' command itself
    coherence_parser = subparser.add_parser(
        "coherence",
        help="Run coherence evaluation tasks (training or evaluation).",
        description="Tools for training or evaluating sequence coherence models.",
        # Add aliases if desired: aliases=('coh',)
    )

    # 2. Add the subparsers action *to the coherence_parser*
    # This enables 'coherence train' and 'coherence eval'
    coherence_subparsers = coherence_parser.add_subparsers(
        dest="coherence_command",  # Destination for 'train' or 'eval'
        required=True,
        title="Coherence Actions",
        help="Specify action: 'train' or 'eval'.",
    )

    # 3. Call the functions to add each specific subcommand
    # Pass the 'coherence_subparsers' object and 'common_parser' down
    add_train_subcommand(subparsers=coherence_subparsers, common_parser=common_parser)
    add_eval_subcommand(subparsers=coherence_subparsers, common_parser=common_parser)

    # Note: No set_defaults here for coherence_parser itself, unless 'coherence'
    # could be run without 'train' or 'eval', which is not the case since
    # coherence_subparsers is required=True. The func comes from the final subcommand.
