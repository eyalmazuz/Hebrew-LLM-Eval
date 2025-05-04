import pandas as pd
import argparse
from src.utils import load_model, StanceDataset, evaluate_model, predict_stance
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import KFold
from tqdm import tqdm


def run_cross_validation(
    texts, 
    labels, 
    model_name, 
    n_splits=5, 
    batch_size=8, 
    random_state=42,
    display_fold_results=True
):
    """
    Run k-fold cross-validation on the stance classification model.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        model_name: Path to the model directory
        n_splits: Number of folds for cross-validation
        batch_size: Batch size for evaluation
        random_state: Random seed for reproducibility
        display_fold_results: Whether to display results for each fold
        
    Returns:
        Dictionary of average performance metrics across all folds
    """
    # Create dynamic label mapping from the entire dataset
    unique_labels = sorted(set(labels))
    LABEL2ID = {label: idx for idx, label in enumerate(unique_labels)}
    ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
    
    # Load model and tokenizer once
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Metrics collectors
    all_accuracies = []
    all_f1_scores = []
    all_precisions = []
    all_recalls = []
    
    # Run cross-validation
    print(f"Running {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(texts), total=n_splits)):
        # Split data for this fold
        fold_train_texts = [texts[i] for i in train_idx]
        fold_train_labels = [labels[i] for i in train_idx]
        fold_test_texts = [texts[i] for i in test_idx]
        fold_test_labels = [labels[i] for i in test_idx]
        
        # Create datasets
        test_dataset = StanceDataset(fold_test_texts, fold_test_labels, tokenizer, LABEL2ID)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Evaluate
        model.eval()
        predictions, references = [], []
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().tolist())
                references.extend(batch["labels"].cpu().tolist())
        
        # Calculate metrics
        accuracy = load("accuracy").compute(predictions=predictions, references=references)
        f1_macro = load("f1").compute(predictions=predictions, references=references, average="macro")
        precision = load("precision").compute(predictions=predictions, references=references, average="macro", zero_division=0)
        recall = load("recall").compute(predictions=predictions, references=references, average="macro")
        
        # Store results
        all_accuracies.append(accuracy["accuracy"])
        all_f1_scores.append(f1_macro["f1"])
        all_precisions.append(precision["precision"])
        all_recalls.append(recall["recall"])
        
        if display_fold_results:
            print(f"\nFold {fold+1} results:")
            print({
                "Accuracy": accuracy["accuracy"], 
                "F1 macro": f1_macro["f1"], 
                "Precision": precision["precision"], 
                "Recall": recall["recall"]
            })
    
    # Calculate and return average metrics
    avg_results = {
        "Accuracy": np.mean(all_accuracies),
        "Accuracy_std": np.std(all_accuracies),
        "F1_macro": np.mean(all_f1_scores),
        "F1_macro_std": np.std(all_f1_scores),
        "Precision": np.mean(all_precisions),
        "Precision_std": np.std(all_precisions),
        "Recall": np.mean(all_recalls),
        "Recall_std": np.std(all_recalls)
    }
    
    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation test.")
    parser.add_argument("--data", type=str, required=True, help="Dataset path (e.g., ./Data/Hebrew_stance_dataset_combined.csv).")
    parser.add_argument("--model", type=str, required=True, help="Model directory path.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--no_fold_results", action='store_true', help="Don't display individual fold results.")
    args = parser.parse_args()
    
    try:
        path_to_csv = args.data
        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"Dataset file not found: {path_to_csv}")
            
        df = pd.read_csv(path_to_csv)
        print(f"Dataset loaded successfully. Total samples: {len(df)}")

        # Get data
        texts = df["Text"].tolist()
        labels = df["Topic"].tolist()

        # Run cross-validation
        cv_results = run_cross_validation(
            texts=texts,
            labels=labels,
            model_name=args.model,
            n_splits=args.folds,
            batch_size=args.batch_size,
            display_fold_results=not args.no_fold_results
        )
        
        print("\nCross-Validation Summary:")
        print(f"Number of folds: {args.folds}")
        print(f"Total samples: {len(texts)}")
        
        print("\nAverage Results (mean ± std):")
        for metric in ["Accuracy", "F1_macro", "Precision", "Recall"]:
            mean_value = cv_results[metric]
            std_value = cv_results[f"{metric}_std"]
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()