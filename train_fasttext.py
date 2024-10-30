import argparse
from collections.abc import Generator

from gensim import utils
from gensim.models import FastText
from gensim.utils import tokenize


class DataIter:
    def __init__(self, input_path: str) -> None:
        self.input_path = input_path

    def __iter__(self) -> Generator[list[str], None, None]:
        with utils.open(self.input_path, "r", encoding="utf-8") as fd:
            for i, line in enumerate(fd):
                yield list(tokenize(line))


def train_model(input_path: str, embedding_size: int, window: int, min_count: int, epochs: int) -> FastText:
    model = FastText(vector_size=embedding_size, window=window, min_count=min_count)

    print("Building vocabulary")
    model.build_vocab(corpus_iterable=DataIter(input_path))

    total_examples = model.corpus_count
    print(f"Training using {total_examples} examples")

    model.train(corpus_iterable=DataIter(input_path), total_examples=total_examples, epochs=epochs)
    print("Done Training")

    return model


def main(args: argparse.Namespace) -> None:
    model = train_model(args.input_path, args.embedding_size, args.window, args.min_count, args.epochs)
    print("Saving model")
    model.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        required=True,
        help="Path to txt corpus",
    )
    parser.add_argument("--output-path", "-o", type=str, required=True, help="Path to save the fasttext model")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of training epochs for the model")
    parser.add_argument("--embedding-size", "-es", type=int, default=512, help="Size of the word embedding")
    parser.add_argument("--window", "-w", type=int, default=5, help="Size of the sliding window")
    parser.add_argument(
        "--min-count", "-mc", type=int, default=1, help="Minimum frequency of words for then to included in training"
    )
    args = parser.parse_args()

    main(args)
