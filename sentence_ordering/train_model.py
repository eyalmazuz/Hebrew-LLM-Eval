import argparse
import json
import os
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

idx2source = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}

os.environ["WANDB_PROJECT"] = "Mafat-Coherence"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


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
        train_set = [
            summary
            for summary in summaries
            if summary["metadata"]["source"] != source_type
        ]

        test_set = [
            summary
            for summary in summaries
            if summary["metadata"]["source"] == source_type
        ]

    else:
        raise ValueError(f"Invlid split type was selected {split_type}")

    return train_set, test_set


def generate_permuted_texts(
    text: str, permutation_count: int, block_size: int
) -> list[tuple[str, int]]:
    # Split the text into sentences based on periods
    sentences = text.strip().split(".")

    # Remove any empty sentences (due to trailing periods)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group sentences into blocks of size c
    blocks = [
        ". ".join(sentences[i : i + block_size])
        for i in range(0, len(sentences), block_size)
    ]

    # Generate K unique permutations
    permutations: list[tuple[str, int]] = []
    for _ in range(permutation_count):
        # Shuffle sentences and join them back into a single text
        random.shuffle(blocks)
        permuted_text = ". ".join(sentences) + "."
        permutations.append((permuted_text, 0))

    return permutations


def extract_texts(
    summaries: list[dict[str, Any]],
    include_summaries: bool,
    permutation_count: int,
    block_size: int,
) -> list[tuple[str, int]]:
    data: list[tuple[str, int]] = []
    for summary in summaries:
        if "text_raw" in summary and summary["text_raw"] is not None:
            data.append((summary["text_raw"], 1))  # Add the original with label=1
            data.extend(
                generate_permuted_texts(
                    summary["text_raw"], permutation_count, block_size
                )  # Add the all permutation with label=0
            )

        if (
            "ai_summary" in summary["metadata"]
            and summary["metadata"]["ai_summary"] is not None
        ):
            data.append(
                (summary["metadata"]["ai_summary"], 1)
            )  # Add the original with label=1
            data.extend(
                generate_permuted_texts(
                    summary["metadata"]["ai_summary"], permutation_count, block_size
                )  # Add the all permutation with label=0
            )

        if (
            include_summaries
            and "summary" in summary
            and summary["summary"] is not None
        ):
            data.append((summary["summary"], 1))  # Add the original with label=1
            data.extend(
                generate_permuted_texts(
                    summary["summary"], permutation_count, block_size
                )
            )  # Add the all permutation with label=0

    return data


class SummaryDataset(Dataset):
    def __init__(
        self, data: list[tuple[str, int]], tokenizer: PreTrainedTokenizer
    ) -> None:
        self.texts = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.tokenizer = tokenizer

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True)
        encoding["label"] = float(label)

        return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(eval_pred: EvalPrediction):
    preds = (
        eval_pred.predictions[0]
        if isinstance(eval_pred.predictions, tuple)
        else eval_pred.predictions
    )
    result = multi_label_metrics(predictions=preds, labels=eval_pred.label_ids)
    return result


def main(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.input_path}")
    summaries = load_data(args.input_path)

    source_type = os.environ.get("SLURM_ARRAY_TASK_ID", args.source_type)

    if source_type is None and args.split_type.lower() == "source":
        raise ValueError(
            f"Split type {args.split_type} was chosen but no source was selected"
        )

    print(f"Splitting to train test using {args.split_type}")
    train_summaries, test_summaries = get_train_test_split(
        summaries, args.split_type, source_type, args.test_size
    )

    print("Extracting texts")
    train_data = extract_texts(
        train_summaries, args.include_summaries, args.permutation_count, args.block_size
    )
    test_data = extract_texts(
        test_summaries, args.include_summaries, args.permutation_count, args.block_size
    )

    print("Loading Tokenizer")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Creating Datasets")
    train_dataset = SummaryDataset(train_data, tokenizer)
    test_dataset = SummaryDataset(test_data, tokenizer)

    print(f"Loading model {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        num_labels=1,
    )

    print("Creating training args")
    data_collator = DataCollatorWithPadding(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.save_path,
        eval_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        # tf32=True,
        bf16_full_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Change to accuracy or any other metric
        greater_is_better=False,  # Need to change to True when using accuracy
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        torch_compile=True,
        report_to="wandb",
        run_name=f"{args.model}_{args.split_type}_{args.source_type}".replace("/", "_"),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Path to the summarization data",
    )

    parser.add_argument(
        "-o",
        "--save-path",
        type=str,
        required=True,
        help="Path to save the trained model",
    )

    parser.add_argument(
        "-st",
        "--split-type",
        type=str,
        choices=["random", "source"],
        default="source",
        help="Which type of split to use for the data",
    )

    parser.add_argument(
        "--source-type",
        type=str,
        choices=["Weizmann", "Wikipedia", "Bagatz", "Knesset", "Israel_Hayom"],
        help="Which source to use a test set",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        help="Size of the test set when using random split",
    )

    parser.add_argument(
        "--include-summaries",
        action="store_true",
        help="Whether to use the summaties when training the model or not",
    )

    parser.add_argument(
        "-pc",
        "--permutation-count",
        type=int,
        default=5,
        help="How many permutation to perform for each text instance",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="What blocksize to use when training the model",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="onlplab/alephbert-base",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    # Enforce the condition after parsing
    if args.split_type == "Source" and args.test_size is not None:
        parser.error("Can't set test size when using Source-based split")
    elif args.split_type == "random" and args.test_size is None:
        parser.error(
            "You must mention the size of the test set when using random split"
        )

    main(args)
