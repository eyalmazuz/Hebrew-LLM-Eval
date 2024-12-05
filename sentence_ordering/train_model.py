import argparse
import json
import math
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

import wandb

idx2source = {
    0: "Weizmann",
    1: "Wikipedia",
    2: "Bagatz",
    3: "Knesset",
    4: "Israel_Hayom",
}

os.environ["WANDB_PROJECT"] = "Mafat-Coherence"
os.environ["WANDB_LOG_MODEL"] = "end"


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
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group sentences into blocks of size block_size
    blocks = [
        ". ".join(sentences[i : i + block_size])
        for i in range(0, len(sentences), block_size)
    ]

    original_order = tuple(blocks)
    num_blocks = len(blocks)

    # Compute total number of possible permutations
    total_permutations = math.factorial(num_blocks)
    max_unique_permutations = total_permutations - 1  # Exclude the original order

    # Adjust permutation_count if necessary
    permutation_count = min(permutation_count, max_unique_permutations)

    # Generate random permutations
    permutations_set: set[tuple[str, ...]] = set()
    while len(permutations_set) < permutation_count:
        perm = blocks[:]
        random.shuffle(perm)
        perm_tuple = tuple(perm)
        if perm_tuple != original_order:
            permutations_set.add(perm_tuple)

    return [(". ".join(perm) + ".", 0) for perm in permutations_set]



def extract_texts(
    summaries: list[dict[str, Any]],
    include_summaries: bool,
    only_summaries: bool,
    permutation_count: int,
    block_size: int,
) -> list[tuple[str, int]]:
    data: list[tuple[str, int]] = []
    for summary in summaries:
        if not only_summaries:
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
            (include_summaries or only_summaries)
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
        self,
        data: list[tuple[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.texts = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length != -1 else 8192

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, padding=True, truncation=True, max_length=self.max_length
        )
        encoding["label"] = label

        return encoding


def compute_metrics(eval_pred: EvalPrediction):
    preds = (
        eval_pred.predictions[0]
        if isinstance(eval_pred.predictions, tuple)
        else eval_pred.predictions
    )
    probs = torch.nn.functional.log_softmax(torch.from_numpy(preds), dim=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = eval_pred.label_ids

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {"f1": f1, "roc_auc": roc_auc, "accuracy": accuracy}

    return metrics


def main(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.input_path}")
    summaries = load_data(args.input_path)

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        source_type = idx2source[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        source_type = args.source_type

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
        train_summaries,
        args.include_summaries,
        args.only_summaries,
        args.permutation_count,
        args.block_size,
    )
    test_data = extract_texts(
        test_summaries,
        args.include_summaries,
        args.only_summaries,
        args.permutation_count,
        args.block_size,
    )

    print("Loading Tokenizer")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Creating Datasets")
    train_dataset = SummaryDataset(train_data, tokenizer, args.max_length)
    test_dataset = SummaryDataset(test_data, tokenizer, args.max_length)

    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Test Dataset size: {len(test_dataset)}")

    print(f"Loading model {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        num_labels=2,
        max_position_embeddings=args.max_length if args.max_length != -1 else 512,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )

    print("Creating training args")
    data_collator = DataCollatorWithPadding(tokenizer)

    name = (
        f"Ordering_{args.model}_{args.split_type}_{source_type}_{args.permutation_count}_"
        f"{args.block_size}"
    ).replace("/", "_")

    wandb.init( # type: ignore
        name=name,
        project=os.environ.get("WANDB_PROJECT", None),
        entity=os.environ.get("WANDB_ENTITY", None),
        group="Sentence_Ordering",
        config={
            "source_type": args.source_type,
            "split_type": args.split_type,
            "only_summaries": args.only_summaries,
            "include_summaries": args.include_summaries,
            "permutation_count": args.permutation_count,
            "block_size": args.block_size,
        }
    )

    train_args = TrainingArguments(
        output_dir=f"{args.save_path}/{name}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=10,
        learning_rate=6e-4,
        lr_scheduler_type="cosine",
        # warmup_ratio=0.2,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        tf32=True,
        bf16_full_eval=True,
        # gradient_accumulation_steps=32,
        # gradient_checkpointing=True,
        # load_best_model_at_end=True,
        metric_for_best_model="loss",  # Change to accuracy or any other metric
        greater_is_better=False,  # Need to change to True when using accuracy
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        torch_compile=True,
        report_to="wandb",
        group_by_length=True,
        run_name=name,
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
        help="Whether to include the summaries when training the model",
    )

    parser.add_argument(
        "--only-summaries",
        action="store_true",
        help="Whether to only use the summaries when training the model",
    )

    parser.add_argument(
        "--permutation-count",
        type=int,
        default=5,
        help="How many permutation to perform for each text instance",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=-1,
        help=(
            "the maximum length of test to keep in the training data."
            "This is different than the transformer context length since we avoid truncation"
        ),
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
        default="dicta-il/alephbertgimmel-base",
        help="Path to save the trained model",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="What batch size to use when training the model",
    )

    args = parser.parse_args()

    # Enforce the condition after parsing
    if args.split_type == "Source" and args.test_size is not None:
        parser.error("Can't set test size when using Source-based split")
    elif args.split_type == "random" and args.test_size is None:
        parser.error(
            "You must mention the size of the test set when using random split"
        )
    if args.include_summaries and args.only_summaries:
        parser.error(
            "Yoy must either set only-summaries or include-summaries and not both"
        )

    main(args)
