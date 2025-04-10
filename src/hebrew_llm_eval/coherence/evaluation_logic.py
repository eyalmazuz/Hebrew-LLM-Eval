import torch
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification

from .data.dataset import ShuffleRankingDataset


def ranking_eval(
    test_data: list[str],
    model,
    model_name_or_path: str,
    tokenizer,
    k_max: int,
    max_length: int,
    device: str,
) -> dict[str, float]:
    test_dataset = ShuffleRankingDataset(test_data, k_max, tokenizer=tokenizer, max_length=max_length)

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

    top_correct = 0
    top_total = 0
    pair_correct = 0
    pair_total = 0
    total_mrr = 0.0
    for i in trange(len(test_dataset)):
        encodings, len_shuffled = test_dataset[i]
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Check if evaluation should be skipped based on lack of shuffles
        if len_shuffled == 0:
            print("  --> Skipping this item for ranking (cannot shuffle)")
        else:
            # print("  --> Evaluating this item...")
            with torch.no_grad():
                outputs = model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]

                top_correct += int(torch.max(probs) == probs[0])
                pair_correct += int(torch.sum(probs[0] > probs[1:]))
                pair_total += len_shuffled
                top_total += 1
                all_scores = probs

                # Sort scores in descending order to find the rank
                sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)

                # Find the rank (1-based index) of the original document
                # Find where the index 0 (original document's index) appears in the sorted indices tensor
                rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
                total_mrr += 1.0 / rank

    top_ranking = top_correct / top_total
    pair_ranking = pair_correct / pair_total
    mean_mrr = total_mrr / len(test_dataset)

    print(f"Top ranking accuracy = {top_ranking:.3f}")
    print(f"Pair ranking accuracy = {pair_ranking:.3f}")
    print(f"Pair ranking accuracy = {mean_mrr:.4f}")

    return {"top_ranking_accuracy": top_ranking, "pair_ranking_accuracy": pair_ranking, "mean_mrr": mean_mrr}
