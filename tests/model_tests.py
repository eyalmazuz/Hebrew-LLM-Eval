import pandas as pd
import argparse
from src.utils import load_model, StanceDataset, evaluate_model, predict_stance
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import torch
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer


if __name__ == "__main__":
    # python -m tests.model_tests --data ./Data/Hebrew_stance_dataset_combined.csv -model weighted
    parser = argparse.ArgumentParser(description="Run test.")
    parser.add_argument("--data", type=str, required=True, help="Dataset path (e.g., ./Data/Hebrew_stance_dataset_combined.csv).")
    parser.add_argument("--model", type=str, default="weighted_balanced", help="Method for training (e.g., weighted, balanced, weighted_balanced).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Optional: Index of a specific sample to predict and display.")
    args = parser.parse_args()
    
    try:
        path_to_csv = args.data
        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"Dataset file not found: {path_to_csv}")
            
        df = pd.read_csv(path_to_csv)
        print(f"Dataset loaded successfully. Total samples: {len(df)}")

        # Split data
        texts = df["Text"].tolist()
        labels = df["Topic"].tolist()  

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        eval_texts, test_texts, eval_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(eval_texts)}")
        print(f"Test samples: {len(test_texts)}")

        # Create dynamic label mapping
        unique_labels = sorted(set(labels))
        LABEL2ID = {label: idx for idx, label in enumerate(unique_labels)}
        ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
        print(f"Label mapping: {LABEL2ID}")

        pre = 'fine_tuned_dictabert_topic_stance'
        # model_name = f'{pre}_{args.model}'
        model_name = args.model

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = StanceDataset(train_texts, train_labels, tokenizer, LABEL2ID)
        eval_dataset = StanceDataset(eval_texts, eval_labels, tokenizer, LABEL2ID)
        test_dataset = StanceDataset(test_texts, test_labels, tokenizer, LABEL2ID)


        # Evaluation
        print("\nEvaluating on test set...")
        
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        model.eval()
        predictions, references = [], []

        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().tolist())
                references.extend(batch["labels"].cpu().tolist())


        accuracy = load("accuracy").compute(predictions=predictions, references=references)
        f1_macro = load("f1").compute(predictions=predictions, references=references, average="macro")
        precision = load("precision").compute(predictions=predictions, references=references, average="macro", zero_division=0)
        recall = load("recall").compute(predictions=predictions, references=references, average="macro")

        print("\nEvaluation results:")
        print({
            "Accuracy": accuracy["accuracy"], 
            "F1 macro": f1_macro["f1"], 
            "Precision": precision["precision"], 
            "Recall": recall["recall"]
        })
        
        # confusion matrix
        confusion_matrix = load("confusion_matrix")
        cm_results = confusion_matrix.compute(predictions=predictions, references=references)
        
        print("\nConfusion Matrix:")
        # Create a nice readable format for the confusion matrix
        labels_list = [ID2LABEL[i] for i in range(len(ID2LABEL))]
        print("Labels:", labels_list)
        print("Matrix:")
        for row in cm_results['confusion_matrix']:
            print(row)

        # Single example prediction if requested
        if args.sample_idx is not None:
            if args.sample_idx < len(test_texts):
                sample_text = test_texts[args.sample_idx]
                sample_label = test_labels[args.sample_idx]
                prediction = predict_stance(sample_text, model, tokenizer, ID2LABEL)

                print("\nSample prediction:")
                print(f"Text: {sample_text}")
                print(f"True label: {sample_label}")
                print(f"Predicted label: {prediction}")
            else:
                print(f"Sample index {args.sample_idx} out of range. Max index: {len(test_texts)-1}")


    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


   
    
    
