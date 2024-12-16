import argparse
import os
import tomllib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain

import pandas as pd

from src.augmentations import get_augmentations
from src.utils import extract_texts, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to the summaries data to augment")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to save the augmented summaries")
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        required=True,
        choices=["sentence-removal", "sentence-shuffle"],
        help="Which type of augmentations to use on the texts",
    )
    parser.add_argument(
        "--augmentations-config-path",
        type=str,
        help="Config file for augmentation to change the default behaviour",
    )
    parser.add_argument(
        "--max-examples", type=int, default=-1, help="The maximum number of examples to generate augmentatios for"
    )

    parser.add_argument(
        "--num-processes", type=int, default=-1, help="The number of threads to use when running augmentations"
    )

    return parser.parse_args()


def augment_text(text: str, augmentation) -> dict[str, str] | None:
    augmented_text = augmentation(text)
    if augmented_text is not None:
        return {"original-text": text, "augmented-text": augmented_text, "augmentation": str(augmentation)}
    else:
        return None


def augment_text_multiple(text: str, augmentations) -> list[dict[str, str]]:
    augmented_texts: list[dict[str, str]] = []
    for augmentation in augmentations:
        augmneted_text = augment_text(text, augmentation)
        if augmneted_text is not None:
            augmented_texts.append(augmneted_text)

    return augmented_texts


def main(args: argparse.Namespace) -> None:
    summaries = load_data(args.input_path)
    texts = extract_texts(summaries=summaries, only_summaries=True, use_ai_summaries=False)
    if args.max_examples != -1:
        texts = texts[: args.max_examples]

    augmentations_config = None
    if args.augmentations_config_path is not None:
        with open(args.augmentations_config_path, "rb") as fd:
            augmentations_config = tomllib.load(fd)

    augmentations = get_augmentations(args.augmentations, augmentations_config)

    augment_text_partial = partial(augment_text_multiple, augmentations=augmentations)

    results: Iterator[list[dict[str, str]]]
    if args.num_processes == -1:
        results = map(augment_text_partial, texts)

    else:
        print("Running multi-processing")
        with ProcessPoolExecutor(max_workers=args.num_processes) as pool:
            results = pool.map(augment_text_partial, texts, chunksize=10)

    df = pd.DataFrame(chain.from_iterable(results), columns=["original-text", "augmented-text", "augmentation"])
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    df.to_csv(os.path.join(args.output_path, "augmented_texts.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
