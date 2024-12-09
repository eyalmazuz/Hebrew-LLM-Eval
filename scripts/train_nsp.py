import argparse
import os

from transformers import (
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from src.data_utils import generate_negative_pairs, generate_nsp_pairs
from src.datasets import PairDataset
from src.train_utils import compute_metrics
from src.utils import IDX2SOURCE, extract_texts, get_train_test_split, load_data

os.environ["WANDB_PROJECT"] = "Mafat-Coherence"
os.environ["WANDB_LOG_MODEL"] = "end"


def main(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.input_path}")
    summaries = load_data(args.input_path)

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        source_type = IDX2SOURCE[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        source_type = args.source_type

    if source_type is None and args.split_type.lower() == "source":
        raise ValueError(f"Split type {args.split_type} was chosen but no source was selected")

    print(f"Splitting to train test using {args.split_type}")
    train_summaries, test_summaries = get_train_test_split(summaries, args.split_type, source_type, args.test_size)

    print(
        f"Generating Training data with only summaries = {args.only_summaries}, "
        f"negative_count count={args.negative_count}, negative type={args.negative_type}"
    )
    train_texts = extract_texts(
        train_summaries,
        args.only_summaries,
    )
    train_positives = generate_nsp_pairs(train_texts)
    train_negatives = generate_negative_pairs(
        train_texts, negative_count=args.negative_count, negative_type=args.negative_type
    )
    train_data = train_positives + train_negatives

    print(
        f"Generating Test data with only summaries = {args.only_summaries}, "
        f"negative_count count={args.negative_count}, negative type={args.negative_type}"
    )
    test_texts = extract_texts(
        test_summaries,
        args.only_summaries,
    )
    test_positives = generate_nsp_pairs(test_texts)
    test_negatives = generate_negative_pairs(
        test_texts, negative_count=args.negative_count, negative_type=args.negative_type
    )
    test_data = test_positives + test_negatives

    print("Loading Tokenizer")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Creating Datasets")
    train_dataset = PairDataset(train_data, tokenizer, args.max_length)
    test_dataset = PairDataset(test_data, tokenizer, args.max_length)

    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Test Dataset size: {len(test_dataset)}")

    print(f"Loading model {args.model}")
    model = AutoModelForNextSentencePrediction.from_pretrained(
        args.model,
        # max_position_embeddings=args.max_length if args.max_length != -1 else 512,
        ignore_mismatched_sizes=True,
    )

    print("Creating training args")
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    name = (f"NSP_{args.model}_{args.split_type}_{source_type}_{args.negative_count}").replace("/", "_")

    wandb.init(  # type: ignore
        name=name,
        project=os.environ.get("WANDB_PROJECT", None),
        entity=os.environ.get("WANDB_ENTITY", None),
        group="NSP",
        config={
            "source_type": args.source_type,
            "split_type": args.split_type,
            "negative_count": args.negative_count,
            "negative_type": args.negative_type,
            "only_summaries": args.only_summaries,
        },
    )

    train_args = TrainingArguments(
        output_dir=f"{args.save_path}/{name}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=10,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        # bf16=True,
        # tf32=True,
        # bf16_full_eval=True,
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
        "--negative-count",
        type=int,
        default=5,
        help="How many negatives to generate for each pair",
    )

    parser.add_argument(
        "--negative-type",
        type=str,
        default="in-pair",
        choices=["in-pair", "any"],
        help="type of negatives to generate for each pair",
    )

    parser.add_argument(
        "--only-summaries",
        action="store_true",
        help="Whether or not to use only summaries for the training",
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
        parser.error("You must mention the size of the test set when using random split")
    main(args)
