import torch
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification

from .data.dataset import ShuffleRankingDataset


def ranking_eval(
    test_data: list[str], model, model_name_or_path: str, k_max: int, max_length: int, device: str, wandb=None
):
    test_dataset = ShuffleRankingDataset(test_data, k_max, tokenizer_name=model_name_or_path, max_length=max_length)

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            # max_position_embeddings=args.max_length if args.max_length != -1 else 512,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
    model.eval()

    total = 0
    correct = 0
    for i in trange(len(test_dataset)):
        encodings, len_shuffled = test_dataset[i]
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Check if evaluation should be skipped based on lack of shuffles
        if len_shuffled == 0:
            print("  --> Skipping this item for ranking (cannot shuffle)")
        else:
            print("  --> Evaluating this item...")
            with torch.no_grad():
                outputs = model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                is_success = torch.max(probs) == probs[0]
                total += 1
                if is_success:
                    correct += 1
    print(f"Ranking accuracy = {correct / total:.3f}")

    if wandb is not None:
        wandb.summary["ranking_accuracy"] = correct / total  # type: ignore
