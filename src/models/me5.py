from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def create_matching_matrix_with_e5(source_sentences, target_sentences):
    """
    Create a matching matrix using sentence embeddings from multilingual-e5-base.
    """
    if not source_sentences or not target_sentences:
        raise ValueError("Both source_sentences and target_sentences must be non-empty lists.")

    # Load model and tokenizer
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def encode_sentences(sentences):
        """
        Encode sentences into embeddings using the multilingual-e5-base model.
        """
        if not sentences:
            return np.array([])
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling for sentence embeddings

    # Encode source and target sentences
    source_embeddings = encode_sentences(source_sentences)
    target_embeddings = encode_sentences(target_sentences)

    if source_embeddings.size == 0 or target_embeddings.size == 0:
        raise ValueError("Failed to compute embeddings. Ensure input sentences are valid.")

    # Compute cosine similarity matrix
    matching_matrix = cosine_similarity(target_embeddings, source_embeddings)

    # Normalize the matrix if possible
    if matching_matrix.max(axis=1).all():
        normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
    else:
        normalized_matrix = matching_matrix

    return source_sentences, target_sentences, matching_matrix, normalized_matrix


def create_matching_matrix_with_e5_instruct(source_sentences, target_sentences):
    """
    Create a matching matrix using sentence embeddings from multilingual-e5-base-instruct.
    """
    if not source_sentences or not target_sentences:
        raise ValueError("Both source_sentences and target_sentences must be non-empty lists.")

    # Load instruct model and tokenizer
    model_name = "intfloat/multilingual-e5-large-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def encode_sentences(sentences, prefix):
        """
        Encode sentences with instruct prefixes and return embeddings.
        """
        prefixed = [f"{prefix} {s}" for s in sentences]
        inputs = tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Encode sentences with appropriate prompts
    source_embeddings = encode_sentences(source_sentences, "passage:")
    target_embeddings = encode_sentences(target_sentences, "query:")

    if source_embeddings.size == 0 or target_embeddings.size == 0:
        raise ValueError("Failed to compute embeddings. Ensure input sentences are valid.")

    # Compute cosine similarity matrix
    matching_matrix = cosine_similarity(target_embeddings, source_embeddings)

    # Normalize rows
    row_max = matching_matrix.max(axis=1, keepdims=True)
    normalized_matrix = matching_matrix / np.where(row_max == 0, 1, row_max)

    return source_sentences, target_sentences, matching_matrix, normalized_matrix


