import os
import time
import torch
import argparse

from utilities.utilities import *

from modeling.training import *
from modeling.data_wrangler import * 

from quality.quality_vegetation import * 
from quality.quality_geographical_tiles import * 

def _build_parser() -> argparse.ArgumentParser:
    """
    Builds and return the command-line argument parser for main.py.

    This parser defines a single optional positional argument, "task",
    that selects exactly one pipeline stage to execute per run.

    Behavior:
    - If "task" is not provided, the default stage is "training".
    - Valid values for "task" are restricted via "choices" to prevent typos.
    - "trainning" is accepted as an alias for "training".

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """
    p = argparse.ArgumentParser(prog="main.py")

    p.add_argument(
        "task",
        nargs="?",
        default="training",
        choices=["geo", "pasture", "wrangler", "training", "trainning"],
        help=(
            "Pipeline stage to run. "
            "Valid: geo, pasture, wrangler, training, trainning. "
            "Default: training."
        ),
    )
    return p


def main_cli():
    """
    CLI entrypoint that dispatches to exactly one pipeline stage.

    Dispatch rules:
    - task == "geo" runs main_vegetation()
    - task == "pasture" runs main_pasture()
    - task == "wrangler" runs main_data_wrangler()
    - task == "training" runs main_training()

    Notes:
    - This design intentionally avoids nested execution. One run executes one stage.
    - To run multiple stages, invoke main.py multiple times (one stage per run).
    """
    args = _build_parser().parse_args()

    if args.task == "geo":
        main_vegetation()
        return

    if args.task == "pasture":
        main_pasture()
        return

    if args.task == "wrangler":
        main_data_wrangler()
        return

    # "training" (default) or alias "trainning"
    main_training()


if __name__ == "__main__":
    main_cli()
