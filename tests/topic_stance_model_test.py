import pandas as pd
import argparse
from src.utils import load_model, StanceDataset, evaluate_model, predict_stance
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict
import numpy as np

if __name__ == "__main__":
    # python -m tests.model_tests --data ./Data/topic_stance_dataset_combined_shuffled.csv -model stance_detection_model_combined
    parser = argparse.ArgumentParser(description="Run test.")
    parser.add_argument("--data", type=str, required=True, help="Dataset path (e.g., ./Data/topic_stance_dataset_combined_shuffled.csv).")
    parser.add_argument("--model", type=str, default="stance_detection_model_combined", help="Model to test.")
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
        sentences = df["sentence"].tolist()
        topics = df["topic"].tolist()  
        labels = df["stance"].tolist()

        data = list(zip(sentences, topics, labels))

        # Split data
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[x[2] for x in data])
        eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[x[2] for x in temp_data])

        train_sentences, train_topics, train_labels = zip(*train_data)
        eval_sentences, eval_topics, eval_labels = zip(*eval_data)
        test_sentences, test_topics, test_labels = zip(*test_data)

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(eval_data)}")
        print(f"Test samples: {len(test_data)}")

        # Create dynamic label mapping
        unique_labels = sorted(set(labels))
        LABEL2ID = {label: idx for idx, label in enumerate(unique_labels)}
        ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
        print(f"Label mapping: {LABEL2ID}")

        pre = 'fine_tuned_dictabert_topic_stance'
        model_name = args.model

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Preprocessing function (matching your training setup)
        def preprocess(example, tokenizer, label2id):
            combined = f"{example['sentence']} [SEP] {example['topic']}"
            inputs = tokenizer(
                combined,
                truncation=True,
                padding="max_length",
                max_length=128
            )
            inputs["label"] = label2id[example["stance"]]
            return inputs

        # Create datasets
        train_dataset = Dataset.from_dict({"sentence": train_sentences, "topic": train_topics, "stance": train_labels})
        eval_dataset = Dataset.from_dict({"sentence": eval_sentences, "topic": eval_topics, "stance": eval_labels})
        test_dataset = Dataset.from_dict({"sentence": test_sentences, "topic": test_topics, "stance": test_labels})

        dataset_dict = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset})

        # Apply preprocessing with correct scope for label2id (matching your training)
        preprocess_with_args = lambda example: preprocess(example, tokenizer, LABEL2ID)
        tokenized_datasets = dataset_dict.map(preprocess_with_args, batched=False)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]

        # Remove original columns and keep only model inputs
        train_dataset = train_dataset.remove_columns(["sentence", "topic", "stance"])
        eval_dataset = eval_dataset.remove_columns(["sentence", "topic", "stance"])
        test_dataset = test_dataset.remove_columns(["sentence", "topic", "stance"])

        # Set format for PyTorch
        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")
        test_dataset.set_format("torch")

        # Evaluation
        print("\nEvaluating on test set...")
        
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        model.eval()
        predictions, references = [], []
        prediction_probs = []  # Store probabilities for AUC-ROC

        with torch.no_grad():
            for batch in test_dataloader:
                # Separate labels from inputs
                labels = batch.pop("label")  # Remove labels from batch
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = labels.to(model.device)
                
                outputs = model(**batch)
                
                # Get predicted classes
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().tolist())
                references.extend(labels.cpu().tolist())
                
                # Get prediction probabilities for AUC-ROC
                probs = F.softmax(outputs.logits, dim=-1)
                prediction_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays for easier handling
        prediction_probs = np.array(prediction_probs)
        references_array = np.array(references)

        # Calculate standard metrics
        accuracy = load("accuracy").compute(predictions=predictions, references=references)
        f1_macro = load("f1").compute(predictions=predictions, references=references, average="macro")
        precision = load("precision").compute(predictions=predictions, references=references, average="macro", zero_division=0)
        recall = load("recall").compute(predictions=predictions, references=references, average="macro")

        # Calculate AUC-ROC for multi-class
        try:
            # One-vs-Rest AUC-ROC (macro average)
            auc_macro = roc_auc_score(references_array, prediction_probs, multi_class='ovr', average='macro')
            
            # One-vs-Rest AUC-ROC (weighted average)
            auc_weighted = roc_auc_score(references_array, prediction_probs, multi_class='ovr', average='weighted')
            
            # Per-class AUC-ROC scores
            auc_per_class = roc_auc_score(references_array, prediction_probs, multi_class='ovr', average=None)
            
        except Exception as e:
            print(f"Warning: Could not compute AUC-ROC: {e}")
            auc_macro = None
            auc_weighted = None
            auc_per_class = None

        print("\nEvaluation results:")
        results = {
            "Accuracy": accuracy["accuracy"], 
            "F1 macro": f1_macro["f1"], 
            "Precision": precision["precision"], 
            "Recall": recall["recall"]
        }
        
        if auc_macro is not None:
            results["AUC-ROC (macro)"] = auc_macro
            results["AUC-ROC (weighted)"] = auc_weighted
        
        print(results)
        
        # Print per-class AUC-ROC scores
        if auc_per_class is not None:
            print("\nPer-class AUC-ROC scores:")
            for i, (label, auc_score) in enumerate(zip(unique_labels, auc_per_class)):
                print(f"  {label}: {auc_score:.4f}")
        
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
            if args.sample_idx < len(test_data):
                sample_sentence, sample_topic, sample_label = test_data[args.sample_idx]
                combined = f"{sample_sentence} [SEP] {sample_topic}"
                
                # Tokenize the sample (matching your training setup)
                inputs = tokenizer(
                    combined,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
                    predicted_label = ID2LABEL[predicted_id]

                print("\nSample prediction:")
                print(f"Sentence: {sample_sentence}")
                print(f"Topic: {sample_topic}")
                print(f"True label: {sample_label}")
                print(f"Predicted label: {predicted_label}")
                
                # Show prediction probabilities for this sample
                sample_probs = probs.cpu().numpy()[0]
                print("Prediction probabilities:")
                for label, prob in zip(unique_labels, sample_probs):
                    print(f"  {label}: {prob:.4f}")
            else:
                print(f"Sample index {args.sample_idx} out of range. Max index: {len(test_data)-1}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()