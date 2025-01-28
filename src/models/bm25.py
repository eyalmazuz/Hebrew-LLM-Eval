import torch
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

"""
BM25 analysis
For each (source, summary) pair:
  – Break both the source and summary into sentences (documents).
  – Compute the BM25 matrix.
  – Calculate the cosine similarity between the source and summary documents.
"""


def create_matching_matrix_with_bm25_and_cosine(source_sentences, target_sentences, use_cosine=False):
    """
    Create a matching matrix using BM25 similarity scores, with optional cosine similarity.
    """
    if not source_sentences or not target_sentences:
        raise ValueError("Both source_sentences and target_sentences must be non-empty lists.")

    # Tokenize the source sentences
    tokenized_source = [sent.split() for sent in source_sentences]
    bm25 = BM25Okapi(tokenized_source)  # Initialize BM25 with source sentences

    # Initialize the matching matrix
    matching_matrix = np.zeros((len(target_sentences), len(source_sentences)))

    # Compute BM25 scores for each target sentence against source sentences
    for i, target_sent in enumerate(target_sentences):
        tokenized_target = target_sent.split()
        bm25_scores = bm25.get_scores(tokenized_target)
        matching_matrix[i, :] = bm25_scores

    # Normalize the matrix if possible
    if matching_matrix.max(axis=1).all():
        normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
    else:
        normalized_matrix = matching_matrix

    if use_cosine:
        # Compute cosine similarity between normalized BM25 vectors
        cosine_matrix = cosine_similarity(normalized_matrix)
        return source_sentences, target_sentences, matching_matrix, normalized_matrix, cosine_matrix

    return source_sentences, target_sentences, matching_matrix, normalized_matrix


# this function works without the llm splitting
# def create_matching_matrix_with_bm25_and_cosine(source_sentences, target_sentences, use_cosine=False):
#     """
#     Create a matching matrix using BM25 similarity scores, with optional cosine similarity.
#     """
#     # Tokenize the source sentences
#     tokenized_source = [sent.split() for sent in source_sentences]
#     bm25 = BM25Okapi(tokenized_source)  # Initialize BM25 with source sentences
#
#     # Create an empty matching matrix
#     matching_matrix = np.zeros((len(target_sentences), len(source_sentences)))
#
#     # Compute BM25 scores for each target sentence against source sentences
#     for i, target_sent in enumerate(target_sentences):
#         tokenized_target = target_sent.split()
#         bm25_scores = bm25.get_scores(tokenized_target)
#         matching_matrix[i, :] = bm25_scores
#
#     # Normalize the matrix for easier interpretation
#     normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
#
#     if use_cosine:
#         # Compute cosine similarity between normalized BM25 vectors
#         cosine_matrix = cosine_similarity(normalized_matrix)
#         return source_sentences, target_sentences, matching_matrix, normalized_matrix, cosine_matrix
#
#     return source_sentences, target_sentences, matching_matrix, normalized_matrix
