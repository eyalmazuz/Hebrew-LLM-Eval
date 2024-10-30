import argparse

from gensim import utils
from gensim.corpora import WikiCorpus


def get_texts(wiki):
    for doc in wiki.getstream():
        yield [word for word in utils.to_unicode(doc).lower().split()]


def create_wiki_corpus(input_path: str, output_file: str) -> None:
    i = 0

    print("Starting to create wiki corpus")

    with open(output_file, "w", encoding="utf-8") as fd:
        wiki = WikiCorpus(input_path, dictionary={})

        for text in get_texts(wiki):
            article = " ".join([t for t in text])  # Use " ".join instead of space.join

            fd.write(article + "\n")
            i += 1
            if i % 1000 == 0:
                print("Saved " + str(i) + " articles")

    print("Finished - Saved " + str(i) + " articles")


def main(args) -> None:
    create_wiki_corpus(args.input_path, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        required=True,
        default="data/hewiki-latest-pages-articles.xml.bz2",
        help="Path to the wikipedia corpus",
    )
    parser.add_argument(
        "--output-path", "-o", type=str, required=True, default="data/wiki.he.text", help="Path to save the output"
    )
    args = parser.parse_args()

    main(args)
