import argparse

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
        choices=["sentence-removal", "middle-shuffle"],
        help="Which type of augmentations to use on the texts",
    )
    parser.add_argument(
        "--augmentations-config",
        type=str,
        help="Config file for augmentation to change the default behaviour",
    )
    parser.add_argument(
        "--max-examples", type=int, default=-1, help="The maximum number of examples to generate augmentatios for"
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    summaries = load_data(args.input_path)
    texts = extract_texts(summaries=summaries, only_summaries=True, use_ai_summaries=False)
    if args.max_examples != -1:
        texts = texts[: args.max_examples]
    augmentations = get_augmentations(args.augmentations, args.augmentations_config)

    augmented_texts: list[str] = []
    for text in texts:
        for augmentation in augmentations:
            augmented_text = augmentation(text)
            if augmented_text is not None:
                augmented_texts.append(augmented_text)

    with open(args.output_path, "w") as fd:
        fd.write("\n\n\n".join(augmented_texts))  # TODO: change the way texts are saved to the disk


if __name__ == "__main__":
    args = parse_args()
    main(args)
