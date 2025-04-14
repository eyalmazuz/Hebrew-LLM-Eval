import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data.utils import load_data
from ..evaluation_logic import ranking_eval


def run_evaluation(
    # Data/Split Args
    data_path: str,
    # Model/Tokenization Args
    model_name: str,
    max_length: int,
    k_max: int,
    # Training Hyperparameters
    batch_size: int,
    num_labels: int,
    output_dir: str,
    device: str,
):
    # Using your actual load_data function
    print(f"Loading data from: {data_path}")
    texts = load_data(data_path)
    if not texts:
        print("Error: No data loaded. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using your actual ShuffleDataset

    print(f"Loading model {model_name} with {num_labels=}")
    # Use the num_labels argument passed to the function
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # Use the passed argument
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",  # Assuming this remains constant
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
        if "cuda" in device and torch.cuda.is_bf16_supported()
        else torch.float32,  # Check bf16 support
    ).to(device)

    test_results = ranking_eval(
        test_data=texts,
        model=model,  # Pass the loaded best model
        model_name_or_path=model_name,  # Pass path for potential reloading if needed
        tokenizer=tokenizer,
        k_max=k_max,
        max_length=max_length,
        device=device,
        num_labels=num_labels,  # Pass num_labels to ranking_eval if needed
    )
    print(f"Test Set Evaluation Results: {test_results}")

    with open(output_dir, "w") as f:
        json.dump(test_results, f, indent=4)
