from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler, TrainingArguments, Trainer
import evaluate
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from evaluate import load
import numpy as np
import os

LABEL_MAPPING = {"against": 0, "neutral": 1, "support": 2}
REVERSE_LABEL_MAPPING = {0: "against", 1: "neutral", 2: "support"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define a custom dataset
class StanceDataset(Dataset):
    def __init__(self, texts, topics, labels, tokenizer, max_length=512):
        self.texts = texts
        self.topics = topics
        self.labels = [LABEL_MAPPING[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            self.topics[idx],  # Tokenizer will automatically insert <SEP> token
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()} 
        encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding


def load_model(model_name):
    """Load the stance detection model and tokenizer."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model and tokenizer loaded successfully from {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def predict_stance(text, topic, model, tokenizer):
    model.to(device)  # Ensure model is on the correct device
    encoding = tokenizer(
        text, 
        topic, 
        truncation=True, 
        padding="max_length", 
        max_length=512, 
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return REVERSE_LABEL_MAPPING[prediction]


def evaluate_model(model, eval_dataloader):
    model.to(device)  # Move model to GPU/CPU
    model.eval()
    predictions = []
    references = []
    
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move all batch tensors to device
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().tolist())
        references.extend(batch["labels"].cpu().tolist())
    
    metric = load("accuracy")
    return metric.compute(predictions=predictions, references=references)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1).tolist()
    accuracy = load("accuracy").compute(predictions=preds, references=labels)
    f1_macro = load("f1").compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": accuracy["accuracy"], "f1_macro": f1_macro["f1"]}


if __name__ == '__main__':
    try:
        path_to_csv = './Data/Hebrew_stance_dataset.csv'
        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"Dataset file not found: {path_to_csv}")
            
        df = pd.read_csv(path_to_csv)
        print(f"Dataset loaded successfully. Total samples: {len(df)}")

        texts = df["Text"].tolist()
        topics = df["Topic"].tolist()
        labels = df["Stance"].tolist()

        train_texts, temp_texts, train_topics, temp_topics, train_labels, temp_labels = train_test_split(
            texts, topics, labels, test_size=0.2, random_state=42
        )
        eval_texts, test_texts, eval_topics, test_topics, eval_labels, test_labels = train_test_split(
            temp_texts, temp_topics, temp_labels, test_size=0.5, random_state=42
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(eval_texts)}")
        print(f"Test samples: {len(test_texts)}")

        model_name = 'dicta-il/dictabert-sentiment'
        _, tokenizer = load_model(model_name)

        train_dataset = StanceDataset(train_texts, train_topics, train_labels, tokenizer)
        eval_dataset = StanceDataset(eval_texts, eval_topics, eval_labels, tokenizer)
        test_dataset = StanceDataset(test_texts, test_topics, test_labels, tokenizer)

        output_dir = 'fine_tuned_dictabert'
        os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------ OpTuna  ------------------------------------------------------------
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        def hp_space_optuna(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
                "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
            }

        # Default args (will be overridden)
        base_training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False
        )

        print("\nStarting Optuna hyperparameter search...")
        trainer = Trainer(
            model_init=model_init,
            args=base_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space_optuna,
            backend="optuna",
            n_trials=10
        )

        print("Best trial hyperparameters:", best_trial.hyperparameters)

# ------------------------------------------------------------ Training ------------------------------------------------------------
        output_dir = 'fine_tuned_dictabert'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Retrain with best params
        final_training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            **best_trial.hyperparameters
        )

        final_trainer = Trainer(
            model_init=model_init,
            args=final_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        print("\nTraining with best hyperparameters...")
        final_trainer.train()
        final_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save final model
        final_trainer.save_model(output_dir)  # This saves both model and tokenizer
        tokenizer.save_pretrained(output_dir)
        print(f"Training complete. Model saved to {output_dir}")

# ------------------------------------------------------------ Evaluation ------------------------------------------------------------
        print("\nTraining complete. Evaluating on test set...")
        final_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        test_dataloader = DataLoader(test_dataset, batch_size=8)
        test_results = evaluate_model(final_model, test_dataloader)

        print(f"Test Accuracy: {test_results['accuracy']:.4f}")

        sample_idx = 0
        sample_text = test_texts[sample_idx]
        sample_topic = test_topics[sample_idx]
        sample_label = test_labels[sample_idx]
        prediction = predict_stance(sample_text, sample_topic, final_model, tokenizer)

        print("\nSample prediction:")
        print(f"Text: {sample_text}")
        print(f"Topic: {sample_topic}")
        print(f"True stance: {sample_label}")
        print(f"Predicted stance: {prediction}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")



# ------------------------------------------------------------ second training ------------------------------------------------------------
    # # Set up optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )

    # # Detect device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # Training loop
    # print("Starting training...")
    # progress_bar = tqdm(range(num_training_steps))
    # model.train()
    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)
    
    # print("Training complete.")

    # # Save fine-tuned model
    # model.save_pretrained("fine_tuned_dictabert")
    # tokenizer.save_pretrained("fine_tuned_dictabert")

    # # Evaluation
    # metric = evaluate.load("accuracy")
    # model.eval()
    # all_predictions = []
    # all_labels = []

    # for batch in eval_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)

    #     all_predictions.extend(predictions.cpu().numpy())  # Collect predictions
    #     all_labels.extend(batch["labels"].cpu().numpy())  # Collect true labels

    # final_accuracy = metric.compute(predictions=all_predictions, references=all_labels)
    # print(f"Evaluation Accuracy: {final_accuracy['accuracy']:.4f}")

    # # Try prediction
    # text_example = "חיסונים הם דבר מסוכן!"
    # topic_example = "קורונה"

    # stance_prediction = predict_stance(text_example, topic_example, model, tokenizer)
    # print(f"Predicted Stance: {stance_prediction}")
