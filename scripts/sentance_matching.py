import os

from datasets import load_dataset, Dataset
from src.data_loader import load_data
import pandas as pd
from gensim.models import Word2Vec
from src.models.tdidf import create_matching_matrix as match_tfidf
from src.models.me5 import create_matching_matrix_with_e5
from src.models.bm25 import create_matching_matrix_with_bm25_and_cosine
import argparse
import re
import csv
from datetime import datetime


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "./data/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def generate_output_filename(model_name, dataset_name, file_type="results"):
    """Generate standardized output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean up dataset name by removing any path components and special characters
    dataset_name = os.path.basename(dataset_name).replace('/', '_')
    return f"{model_name.lower()}_{dataset_name}_{file_type}_{timestamp}.csv"


def save_results(results_data, model_name, dataset_name, file_type="results"):
    """Save results to CSV in the output directory."""
    output_dir = ensure_output_dir()
    filename = generate_output_filename(model_name, dataset_name, file_type)
    output_path = os.path.join(output_dir, filename)

    try:
        print(f"Saving results to {output_path}...")
        if results_data:
            # Dynamically generate fieldnames from the keys of the first entry
            fieldnames = results_data[0].keys()

            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
        print("Results saved successfully.")
        return output_path
    except Exception as e:
        print(f"Error saving results: {e}")
        return None


def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences


def train_embeddings(dataset):
    """Train word embeddings on the dataset with robust preprocessing."""
    corpus = []

    try:
        # Get total number of items
        dataset_length = get_dataset_length(dataset)
        # print(f"Processing {dataset_length} items for word embeddings...")

        # Process each item in the dataset
        for idx in range(dataset_length):
            try:
                article, summary = process_dataset_item(dataset, idx)

                # Process article
                if isinstance(article, str):
                    sentences = split_into_sentences(article)
                    for sent in sentences:
                        words = re.findall(r'\b\w+\b', sent.lower())  # Convert to lowercase
                        if words:
                            corpus.append(words)

                # Process summary
                if isinstance(summary, str):
                    sentences = split_into_sentences(summary)
                    for sent in sentences:
                        words = re.findall(r'\b\w+\b', sent.lower())  # Convert to lowercase
                        if words:
                            corpus.append(words)

            except Exception as e:
                print(f"Warning: Error processing item {idx}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in train_embeddings: {str(e)}")
        raise

    if not corpus:
        raise ValueError("No valid text found for training word embeddings.")

    # print(f"Training Word2Vec model on {len(corpus)} sentences...")
    model = Word2Vec(sentences=corpus,
                     vector_size=100,
                     window=5,
                     min_count=1,
                     workers=4)

    return model.wv


def process_dataset_item(dataset, idx):
    """Process both HuggingFace and custom dataset formats."""
    if isinstance(dataset, Dataset):
        # For HuggingFace datasets
        article = dataset[idx]['article']
        summary = dataset[idx]['summary']
    elif isinstance(dataset, dict):
        # For dictionary-like datasets
        article = dataset['article'][idx]
        summary = dataset['summary'][idx]
    else:
        raise ValueError(f"Unexpected dataset format: {type(dataset)}")
    return article, summary


def get_dataset_length(dataset):
    """Get length of dataset regardless of its format."""
    if isinstance(dataset, Dataset):
        return len(dataset)
    elif isinstance(dataset, dict):
        return len(dataset['article'])
    else:
        raise ValueError(f"Unexpected dataset format: {type(dataset)}")


def process_and_display_results(dataset, num_articles, match_fn, method, dataset_name):
    """Process and display results for article matching and save to CSV."""
    results_data = []
    metadata_data = []  # For storing metadata about each processed article

    dataset_length = get_dataset_length(dataset)
    num_articles_to_process = min(num_articles, dataset_length)

    print(f"\nArticle Matching Results ({method})")
    print(f"Processing {num_articles_to_process} articles out of {dataset_length} total articles")

    for idx in range(num_articles_to_process):
        print(f"\nArticle {idx + 1}")

        try:
            article, summary = process_dataset_item(dataset, idx)

            source_chunks = split_into_sentences(article)
            target_chunks = split_into_sentences(summary)

            if not source_chunks or not target_chunks:
                print(f"Warning: Empty chunks found for article {idx + 1}, skipping...")
                continue

            results = match_fn(source_chunks, target_chunks)
            source, target, matching_matrix, _ = results

            matching_df = pd.DataFrame(
                matching_matrix,
                index=[f"Target {i + 1}" for i in range(len(target_chunks))],
                columns=[f"Source {j + 1}" for j in range(len(source_chunks))]
            )
            print("Matching Matrix:")
            print(matching_df)

            # Prepare results data
            for i, target_sentence in enumerate(target_chunks):
                best_match_idx = matching_matrix[i].argmax()
                best_match_score = matching_matrix[i, best_match_idx]
                best_match_sentence = source_chunks[best_match_idx]
                results_data.append({
                    "Article": article,
                    "Summary": summary,
                    "Sentence in Summary": target_sentence,
                    "Best Match Sentence From Article": best_match_sentence,
                    "Best Match Score": best_match_score,
                })

            # Prepare metadata
            metadata_data.append({
                "Article_ID": idx,
                "Num_Source_Sentences": len(source_chunks),
                "Num_Target_Sentences": len(target_chunks),
                "Average_Match_Score": matching_matrix.mean(),
                "Max_Match_Score": matching_matrix.max(),
                "Min_Match_Score": matching_matrix.min()
            })

        except Exception as e:
            print(f"Error processing Article {idx + 1}: {str(e)}")
            continue

    # Save results and metadata
    if results_data:
        save_results(results_data, method, dataset_name, "results")
        save_results(metadata_data, method, dataset_name, "metadata")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run article and summary matching.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., biunlp/HeSum or custom).")
    parser.add_argument("--path", type=str, help="Path to custom dataset JSON.")
    parser.add_argument("--model", type=str, required=True, choices=["ME5", "TF-IDF", "BM25"], help="Matching method.")
    parser.add_argument("--num_articles", type=int, default=5, help="Number of articles to process.")
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        ensure_output_dir()

        # Load dataset
        dataset = load_data(args.data, args.path)
        print("Dataset loaded and preprocessed.")

        # Set up the matching function
        if args.model == "TF-IDF":
            print("Training Word Embeddings for TF-IDF...")
            word_embeddings = train_embeddings(dataset)
            print("Embeddings trained successfully.")
            match_fn = lambda src, tgt: match_tfidf(src, tgt, word_embeddings)
        elif args.model == "ME5":
            match_fn = create_matching_matrix_with_e5
        elif args.model == "BM25":
            match_fn = create_matching_matrix_with_bm25_and_cosine

        # Process and display results, and save to CSV
        dataset_name = args.path if args.path else args.data
        process_and_display_results(dataset, args.num_articles, match_fn, args.model, dataset_name)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
