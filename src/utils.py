import json
import random
from typing import Any
import re
import os
from datetime import datetime
import csv
import pandas as pd
from datasets import Dataset
from collections import namedtuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd

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
    metadata_data = []  

    # define namedtuple
    Match = namedtuple('Match', ['summary_sentence', 'article_sentences'])
    Match = namedtuple('Match', [
        'summary_sentence', 
        'article_sentences',
        'stance_preservation', 
        'kl_divergences',
        'summary_stance',
        'summary_stance_score',
        'article_stances',
        'article_stance_scores'
    ])
    matches = []

    dataset_length = get_dataset_length(dataset)
    # num_articles_to_process = min(num_articles, dataset_length)

    print("\nArticle Matching Results")
    # print(f"Processing {num_articles_to_process} articles out of {dataset_length} total articles")

    # for idx in range(dataset_length):
    for idx in range(3):
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

            # matching_df = pd.DataFrame(
            #     matching_matrix,
            #     index=[f"Target {i + 1}" for i in range(len(target_chunks))],
            #     columns=[f"Source {j + 1}" for j in range(len(source_chunks))]
            # )
            # print("Matching Matrix:")
            # print(matching_df)

            # Prepare results data
            for i, target_sentence in enumerate(target_chunks):
                best_matches = []
                # best_matches_scores = []
                for j, source_sentence in enumerate(source_chunks):
                    if matching_matrix[i, j] >= threshold:
                        best_matches.append((source_sentence, matching_matrix[i, j]))
                        # best_matches_scores.append(matching_matrix[i, j])
                best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)[:top_k_matches]
                # best_matches_scores = sorted(best_matches_scores, reverse=True)[:top_k_matches]
                # best_match_idx = matching_matrix[i].argmax()
                # best_match_score = matching_matrix[i, best_match_idx]
                # best_match_sentence = source_chunks[best_match_idx]
                # matches.append(Match(summary_sentence=target_sentence, article_sentences=best_matches))
                match = Match(
                    summary_sentence=target_sentence, 
                    article_sentences=best_matches,
                    stance_preservation=None,  # Default to None
                    kl_divergences=None,       # Default to None
                    summary_stance=None,       # Default to None
                    summary_stance_score=None, # Default to None
                    article_stances=None,      # Default to None
                    article_stance_scores=None # Default to None
                )
                
                matches.append(match)
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

    return matches

# --------------------------------------------------------------- pipeline functions ---------------------------------------------------------------
def load_matching_matrix(csv_path):
    """Load the sentence matching matrix from a CSV file."""
    df = pd.read_csv(csv_path)
    print("Matching Matrix Loaded:")
    print(df.head())
    return df

def load_model(model_name):
    """Load the stance detection model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def classify_stance(sentences, model, tokenizer):
    """Classify stance (sentiment) for a list of sentences and return labels with probabilities."""
    labels = ['negative', 'neutral', 'positive']  # Adjust according to model output labels
    results = []
    probs = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        results.append((labels[predicted_class], probabilities[predicted_class]))
        probs.append(probabilities)
    
    return results, probs

def kl_divergence(p, q):
    """Compute KL-divergence between two probability distributions."""
    p = torch.tensor(p)
    q = torch.tensor(q)
    return F.kl_div(q.log(), p, reduction='sum').item()

def compute_stance_preservation(data, model, tokenizer):
    """Compute stance preservation and KL-divergence between sentiment distributions."""
    updated_data = []
    
    for match in data:
        # Summary sentence
        summary_sentence = match.summary_sentence
        
        # Classify stance for summary sentence
        summary_results, summary_probs = classify_stance([summary_sentence], model, tokenizer)
        summary_stance, summary_score = summary_results[0]
        
        # Process each article sentence
        article_stances = []
        article_stance_scores = []
        article_probs_list = []
        
        for article_sentence in match.article_sentences:
            # Classify stance for each article sentence
            article_results, article_probs = classify_stance([article_sentence[0]], model, tokenizer)
            article_stance, article_score = article_results[0]
            
            article_stances.append(article_stance)
            article_stance_scores.append(article_score)
            article_probs_list.append(article_probs[0])
        
        # Compute KL divergences for each article sentence
        kl_scores = [kl_divergence(summary_probs[0], article_prob) 
                     for article_prob in article_probs_list]
        
        # Determine stance preservation
        # Check if any article sentence has the same stance as the summary
        # TODO change the check
        stance_preserved = 'Preserved' if any(s == summary_stance for s in article_stances) else 'Not Preserved'
        
        # Create a new namedtuple with additional information
        updated_match = match._replace(
            stance_preservation=stance_preserved,
            kl_divergences=kl_scores,
            summary_stance=summary_stance,
            summary_stance_score=summary_score,
            article_stances=article_stances,
            article_stance_scores=article_stance_scores
        )
        
        updated_data.append(updated_match)
    
    return updated_data

    # stance_preservation = []
    # # summary_stance_scores = []
    # # article_stance_scores = []
    # kl_divergences = []
    
    # for idx, row in df.iterrows():
    #     summary_sentence = row['Sentence in Summary']
    #     article_sentence = row['Best Match Sentence From Article']
        
    #     summary_results, summary_probs = classify_stance([summary_sentence], model, tokenizer)
    #     article_results, article_probs = classify_stance([article_sentence], model, tokenizer)

    #     summary_stance, summary_score = summary_results[0]
    #     article_stance, article_score = article_results[0]
        
    #     # summary_stance_scores.append(f"{summary_stance} ({summary_score:.4f})")
    #     # article_stance_scores.append(f"{article_stance} ({article_score:.4f})")
        
    #     """
    #     We want the KL to be small when the summary and article have similar stances.
    #     """
    #     # KL on rows
    #     kl_score = kl_divergence(summary_probs[0], article_probs[0])
    #     kl_divergences.append(kl_score)
        
    #     if summary_stance == article_stance:
    #         stance_preservation.append('Preserved')
    #     else:
    #         stance_preservation.append('Not Preserved')
    
    # # df['Summary Stance (Score)'] = summary_stance_scores
    # # df['Article Stance (Score)'] = article_stance_scores
    # df['Stance Preservation'] = stance_preservation
    # df['KL Divergence'] = kl_divergences
    # return df
