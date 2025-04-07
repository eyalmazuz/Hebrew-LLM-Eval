import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ..data.dataset import ShuffleDataset
from ..data.utils import get_train_test_split, load_data


def compute_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_pred.label_ids

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy}

    return metrics


def run_training(
    data_path: str,
    output_dir: str,
    model_name: str = "dicta-il/alephbertgimmel-base",
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_length: int = 512,
    k_max: int = 20,
    batch_size: int = 32,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    device: str = "cuda",
) -> None:  # Example: return metrics or path to model
    texts = load_data(data_path)
    train_set, test_set = get_train_test_split(texts, test_size)
    train_set, val_set = get_train_test_split(train_set, val_size)
    train_dataset = ShuffleDataset(train_set, k_max, tokenizer_name=model_name, max_length=max_length)
    val_dataset = ShuffleDataset(val_set, k_max, tokenizer_name=model_name, max_length=max_length)

    print(f"Loading model {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        # max_position_embeddings=args.max_length if args.max_length != -1 else 512,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )

    # wandb.init(  # type: ignore
    #     project=os.environ.get("WANDB_PROJECT", None),
    #     entity=os.environ.get("WANDB_ENTITY", None),
    #     group="Sentence_Ordering",
    #     config={
    #         "source_type": args.source_type,
    #         "split_type": args.split_type,
    #         "only_summaries": args.only_summaries,
    #         "permutation_count": args.permutation_count,
    #         "block_size": args.block_size,
    #     },
    # )

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        metric_for_best_model="loss",  # Change to accuracy or any other metric
        greater_is_better=False,  # Need to change to True when using accuracy
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        # torch_compile=True,
        # report_to="wandb",
        # group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=train_dataset.collate,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()
