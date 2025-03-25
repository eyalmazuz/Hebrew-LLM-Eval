import json
import random
from typing import Any
import re
import os
from datetime import datetime
import csv
import pandas as pd
from datasets import Dataset

IDX2SOURCE = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}


def load_data(path: str) -> list[dict[str, Any]]:
    with open(path) as fd:
        summaries = [json.loads(line) for line in fd.readlines()]

    return summaries


def get_train_test_split(
    summaries: list[dict[str, Any]],
    split_type: str,
    source_type: str,
    test_size: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_type.lower() == "random":
        if test_size is not None:
            random.shuffle(summaries)
            train_set = summaries[int(len(summaries) * test_size) :]
            test_set = summaries[: int(len(summaries) * test_size)]
        else:
            raise ValueError("Test size can't be None")

    elif split_type.lower() == "source":
        train_set = [summary for summary in summaries if summary["metadata"]["source"] != source_type]

        test_set = [summary for summary in summaries if summary["metadata"]["source"] == source_type]

    else:
        raise ValueError(f"Invlid split type was selected {split_type}")

    return train_set, test_set


def extract_texts(
    summaries: list[dict[str, Any]],
    only_summaries: bool,
) -> list[str]:
    positives: list[str] = []
    for summary in summaries:
        if (
            not only_summaries
            and "text_raw" in summary
            and summary["text_raw"] is not None
            and summary["text_raw"] != ""
        ):
            positives.append(summary["text_raw"])

        if (
            "ai_summary" in summary["metadata"]
            and summary["metadata"]["ai_summary"] is not None
            and summary["metadata"]["ai_summary"] != ""
        ):
            positives.append(summary["metadata"]["ai_summary"])

        if "summary" in summary and summary["summary"] is not None and summary["summary"] != "":
            positives.append(summary["summary"])

    return list(set(positives))


# --------------------------------------------------------------- sentence matching functions --------------------------------------------------------------- 
def split_into_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        return []
    separators = r"[■|•.\n]"
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]
    return sentences

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "./Data/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def generate_output_filename(dataset_name, file_type="results"):
    """Generate standardized output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean up dataset name by removing any path components and special characters
    dataset_name = os.path.basename(dataset_name).replace('/', '_')
    return f"{dataset_name}_{file_type}_{timestamp}.csv"

def save_results(results_data, dataset_name, file_type="results"):
    """Save results to CSV in the output directory."""
    output_dir = ensure_output_dir()
    filename = generate_output_filename(dataset_name, file_type)
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

def process_and_display_results(dataset, match_fn, dataset_name, save_matches=False, threshold=0.8, top_k_matches=1):
    """Process and display results for article matching and save to CSV."""
    results_data = []
    metadata_data = []  # For storing metadata about each processed article

    dataset_length = get_dataset_length(dataset)
    # num_articles_to_process = min(num_articles, dataset_length)

    print("\nArticle Matching Results")
    # print(f"Processing {num_articles_to_process} articles out of {dataset_length} total articles")

    for idx in range(dataset_length):
        # print(f"\nArticle {idx + 1}")

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
                best_matches = []
                best_matches_scores = []
                for j, source_sentence in enumerate(source_chunks):
                    if matching_matrix[i, j] >= threshold:
                        best_matches.append((source_sentence, matching_matrix[i, j]))
                        best_matches_scores.append(matching_matrix[i, j])
                best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)[:top_k_matches]
                # best_matches_scores = sorted(best_matches_scores, reverse=True)[:top_k_matches]
                # best_match_idx = matching_matrix[i].argmax()
                # best_match_score = matching_matrix[i, best_match_idx]
                # best_match_sentence = source_chunks[best_match_idx]
                results_data.append({
                    "Article": article,
                    "Summary": summary,
                    "Sentence in Summary": target_sentence,
                    "Best Match Sentences From Article": best_matches,
                    # "Best Match Score": best_matches_scores,
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
    if save_matches and results_data:
        save_results(results_data, dataset_name, "results")
        save_results(metadata_data, dataset_name, "metadata")