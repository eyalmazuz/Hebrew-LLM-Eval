from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sentence_transformers import SentenceTransformer


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


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def create_matching_matrix_with_e5_instruct(source_sentences, target_sentences):
    """
    Create a matching matrix using sentence embeddings from multilingual-e5-large-instruct.
    Following the Hugging Face model card example.
    """
    task = "Retrieve semantically similar text."

    if not source_sentences or not target_sentences:
        raise ValueError("Both source_sentences and target_sentences must be non-empty lists.")

    # Prepare queries (target sentences with instruct format)
    queries = [get_detailed_instruct(task, sentence) for sentence in target_sentences]

    # Load instruct model
    model_name = "intfloat/multilingual-e5-large-instruct"
    model = SentenceTransformer(model_name) 

    # input_texts = queries + source_sentences

    # embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
    # num_queries = len(target_sentences)
    # scores = (embeddings[:num_queries] @ embeddings[num_queries:].T) * 100
    
    source_embeddings = model.encode(source_sentences, convert_to_tensor=True, normalize_embeddings=True)
    target_embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)

    scores = target_embeddings @ source_embeddings.T

    return source_sentences, target_sentences, scores.cpu().numpy()

