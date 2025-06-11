# split article and summary into sentences
# use Me5-instruct as the matching method (sentence matching file)
# use dicta model to get the topic for each sentence in the article and summary
# use stance detection model to classify the sentences that contains stances for both article and summary
# classify the stance for each sentence (favor, neutral, against) for both article and summary 
# check stance preservation for each sentence in the article and summary by calculating the difference (or KL-divergence) between the stances of the sentences in the article and summary

import pandas as pd
from src.data_loader import load_data
from src.models.me5 import create_matching_matrix_with_e5_instruct
import argparse
from src.utils import (
    ensure_output_dir, 
    process_and_display_results, 
    compute_stance_preservation_with_topic
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
    )
import torch
from scripts.topic_detection import get_topic_for_model
import numpy as np


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
        match_fn = create_matching_matrix_with_e5_instruct

        # Process and display results, and save to CSV
        dataset_name = args.path if args.path else args.data
        data = process_and_display_results(dataset, match_fn, dataset_name, args.save_matches, args.threshold, args.top_k_matches)
        print(f"Processed {len(data)} matches")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
    
    # Topic detection setup
    print("Setting up topic detection model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  
    )    
    topic_model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm2.0', device_map='cuda', quantization_config=quant_config)
    topic_tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm2.0')
    
    # Fix tokenizer configuration
    if topic_tokenizer.pad_token is None:
        topic_tokenizer.pad_token = topic_tokenizer.eos_token
    
    # Process sentences for topic detection
    print("Processing sentences for topic detection...")
    topic_dataset = []
    total_sentences = len(data)
    
    for idx, sent in enumerate(data):
        try:
            # Get the topic for the current sentence
            if hasattr(sent, 'article_sentences') and sent.article_sentences:
                article_sent = sent.article_sentences[0][0]  # Get first sentence from first match
                summary_sent = sent.summary_sentence

                art_res = get_topic_for_model(article_sent, topic_model, topic_tokenizer)
                sum_res = get_topic_for_model(summary_sent, topic_model, topic_tokenizer)

                # Check topic similarity
                if art_res["topic"] == sum_res["topic"]:
                    res = {
                        "article_sentence": art_res["sentence"],
                        "article_topic": art_res["topic"],
                        "summary_sentence": sum_res["sentence"],
                        "summary_topic": sum_res["topic"]
                    }
                    # print(f"Topic match found: {res['article_topic']}")
                    topic_dataset.append(res)

            # Print progress every 10 sentences
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_sentences} sentences")
                
        except Exception as e:
            print(f"Error processing sentence {idx}: {e}")
            continue

    print(f"Found {len(topic_dataset)} sentence pairs with matching topics")

    if not topic_dataset:
        print("No matching topics found. Exiting.")
        exit(1)

    # Load stance detection model
    print("Loading stance detection model...")
    stance_model_name = './models/stance_detection_model/'
    stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)
    stance_tokenizer = AutoTokenizer.from_pretrained(stance_model_name)
    
     # Classify stance and compute preservation
    print("Computing stance preservation...")
    results = compute_stance_preservation_with_topic(topic_dataset, stance_model, stance_tokenizer)
    
    print(f"Stance preservation computed for {len(results)} sentence pairs")

    # Save results
    if args.save_matches and results:
        output_path = args.output_dir
        
        # Create DataFrame and save as JSON
        result_df = pd.DataFrame(results)
        result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"Stance preservation results saved to {output_path}")
        
        # Print summary statistics
        if len(results) > 0:
            stance_matches = sum(1 for r in results if r['stance_match'])
            avg_emd = np.mean([r['emd_score'] for r in results])
            print(f"Summary: {stance_matches}/{len(results)} stance matches, average EMD: {avg_emd:.4f}")
    else:
        print("No results to save or save_matches flag not set")
     

