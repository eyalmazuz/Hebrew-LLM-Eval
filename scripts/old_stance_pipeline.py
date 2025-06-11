# split article and summary into sentences
# use Me5 as the matching method (sentence matching file)
# use stance detection/sentiment analysis model to classify the sentences that contains stances for both article and summary
# classify the stance for each sentence (favor, neutral, against) for both article and summary 
# check stance preservation for each sentence in the article and summary by calculating the difference (or KL-divergence) between the stances of the sentences in the article and summary

import pandas as pd
from src.data_loader import load_data
from src.models.me5 import create_matching_matrix_with_e5, create_matching_matrix_with_e5_instruct
import argparse
from src.utils import (
    ensure_output_dir, 
    process_and_display_results, 
    load_model, 
    compute_stance_preservation, 
    compute_stance_preservation_emd
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stance pipeline.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., biunlp/HeSum or custom).")
    parser.add_argument("--path", default="./Data/datasets/summarization-7-heb.jsonl", type=str, help="Path to custom dataset JSON.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Matching threshold.")
    parser.add_argument("--top-k-matches", type=int, default=1, help="Number of top matches to consider.")
    parser.add_argument("--save-matches", action="store_true", help="Save matching results to CSV.")
    parser.add_argument("--output-dir", default="./Data/output/stance_preservation_test.json", help="Save stance preservation results to JSON.")
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        ensure_output_dir()

        # Load dataset
        dataset = load_data(args.data, args.path)
        print("Dataset loaded and preprocessed.")

        # match_fn = create_matching_matrix_with_e5
        match_fn = create_matching_matrix_with_e5

        # Process and display results, and save to CSV
        dataset_name = args.path if args.path else args.data
        data = process_and_display_results(dataset, match_fn, dataset_name, args.save_matches, args.threshold, args.top_k_matches)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
    
    # # Load data
    # path_to_csv = 'Data/output/me5_summarization-7-heb.jsonl_results_20250305_195549.csv'
    # df = load_matching_matrix(path_to_csv)
    
    # Load model and tokenizer
    # model_name = 'dicta-il/dictabert-sentiment'
    model_name = 'weighted_test'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Classify stance for each sentence in article and summary
    result = compute_stance_preservation_emd(data, model, tokenizer)
    
    # Save results
    if args.save_matches:
        output_path = args.output_dir
        result_df = pd.DataFrame(result)
        # result_df.to_csv(output_path, index=False)
        result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"Stance preservation results saved to {output_path}")
