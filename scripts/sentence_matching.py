
from src.data_loader import load_data
from src.models.me5 import create_matching_matrix_with_e5
import argparse
from src.utils import ensure_output_dir, process_and_display_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run article and summary matching.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., biunlp/HeSum or custom).")
    parser.add_argument("--path", type=str, help="Path to custom dataset JSON.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Matching threshold.")
    parser.add_argument("--top-k-matches", type=int, default=1, help="Number of top matches to consider.")
    parser.add_argument("--save-matches", action="store_true", help="Save matching results to CSV.")
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        ensure_output_dir()

        # Load dataset
        dataset = load_data(args.data, args.path)
        print("Dataset loaded and preprocessed.")

        match_fn = create_matching_matrix_with_e5

        # Process and display results, and save to CSV
        dataset_name = args.path if args.path else args.data
        process_and_display_results(dataset, match_fn, dataset_name, args.save_matches, args.threshold, args.top_k_matches)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
