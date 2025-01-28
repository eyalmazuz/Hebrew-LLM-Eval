import openai
import torch
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import re
from collections import defaultdict
import streamlit as st


class SoftTfidfVectorizer(TfidfVectorizer):
    """
    Enhanced TF-IDF vectorizer that implements soft term frequency using word embeddings.
    Instead of just counting exact matches, it also counts similar words weighted by
    their embedding similarity: tf = exact_count + sum(similar_word_count * (1 - embedding_distance))
    """

    def __init__(self, word_embeddings, similarity_threshold=0.5, **kwargs):
        """
        Initialize the soft TF-IDF vectorizer.

        Parameters:
        - word_embeddings: KeyedVectors object containing word embeddings
        - similarity_threshold: Minimum similarity for words to be considered related
        """
        super().__init__(**kwargs)
        self.word_embeddings = word_embeddings
        self.similarity_threshold = similarity_threshold

    def _get_embedding_similarity(self, word1, word2):
        """Calculate similarity between two words using embeddings."""
        try:
            if word1 == word2:
                return 1.0
            if word1 in self.word_embeddings and word2 in self.word_embeddings:
                similarity = self.word_embeddings.similarity(word1, word2)
                return max(similarity, 0)  # Ensure non-negative similarity
        except KeyError:
            pass
        return 0.0

    def _compute_soft_term_frequency(self, term, doc_terms):
        """
        Compute soft term frequency that includes similar terms weighted by their similarity.
        tf = exact_count + sum(similar_word_count * (1 - embedding_distance))
        """
        # Start with exact matches
        frequency = doc_terms.count(term)

        # Add weighted counts from similar terms
        if term in self.word_embeddings:
            unique_terms = set(doc_terms)
            for other_term in unique_terms:
                if other_term != term:
                    similarity = self._get_embedding_similarity(term, other_term)
                    if similarity > self.similarity_threshold:
                        # Add weighted count: count * (1 - distance)
                        # Since similarity = 1 - distance, we can use similarity directly
                        frequency += doc_terms.count(other_term) * similarity

        return frequency

    def fit_transform(self, raw_documents, y=None):
        """Fit the vectorizer and transform the documents."""
        # First fit normally to get vocabulary
        result = super().fit(raw_documents)

        # Now transform with soft term frequency
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform documents using soft term frequencies."""
        # Get vocabulary
        vocabulary = self.vocabulary_

        # Initialize matrix
        X = np.zeros((len(raw_documents), len(vocabulary)))

        # Process each document
        for doc_idx, doc in enumerate(raw_documents):
            doc_terms = self.build_analyzer()(doc)

            # Compute soft term frequencies
            term_freqs = defaultdict(float)
            for term in set(doc_terms):
                if term in vocabulary:
                    term_freqs[term] = self._compute_soft_term_frequency(term, doc_terms)

            # Update matrix
            for term, freq in term_freqs.items():
                X[doc_idx, vocabulary[term]] = freq

        # Apply IDF weights
        if self.use_idf:
            X = X * self.idf_

        # Normalize if requested
        if self.norm is not None:
            X = normalize(X, norm=self.norm, axis=1, copy=False)

        return X


# def create_matching_matrix(source_sentences, target_sentences, word_embeddings):
#     """
#     Create a matching matrix using soft TF-IDF similarity scores.
#     """
#     if not source_sentences or not target_sentences:
#         raise ValueError("Both source_sentences and target_sentences must be non-empty lists.")
#
#     # Initialize vectorizer with soft term frequency
#     vectorizer = SoftTfidfVectorizer(
#         word_embeddings=word_embeddings,
#         similarity_threshold=0.5,
#         analyzer='word',
#         token_pattern=r'\\b\\w+\\b'
#     )
#
#     # Combine sentences for vectorization
#     all_sentences = source_sentences + target_sentences
#     tfidf_matrix = vectorizer.fit_transform(all_sentences)
#
#     # Separate source and target matrices
#     source_matrix = tfidf_matrix[:len(source_sentences)]
#     target_matrix = tfidf_matrix[len(source_sentences):]
#
#     # Compute similarity matrix
#     matching_matrix = cosine_similarity(target_matrix, source_matrix)
#
#     # Normalize the matrix if possible
#     if matching_matrix.max(axis=1).all():
#         normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
#     else:
#         normalized_matrix = matching_matrix
#
#     return source_sentences, target_sentences, matching_matrix, normalized_matrix


# this function works without the llm splitting
def create_matching_matrix(source_sentences, target_sentences, word_embeddings):
    """
    Create a matching matrix using soft TF-IDF similarity scores.
    """
    # Initialize vectorizer with soft term frequency
    vectorizer = SoftTfidfVectorizer(
        word_embeddings=word_embeddings,
        similarity_threshold=0.5,
        analyzer='word',
        token_pattern=r'\b\w+\b'
    )

    # Combine sentences for vectorization
    all_sentences = source_sentences + target_sentences
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    # Separate source and target matrices
    source_matrix = tfidf_matrix[:len(source_sentences)]
    target_matrix = tfidf_matrix[len(source_sentences):]

    # Compute similarity matrix
    matching_matrix = cosine_similarity(target_matrix, source_matrix)

    # Normalize for easier interpretation
    normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)

    return source_sentences, target_sentences, matching_matrix, normalized_matrix


# Function to split text into sentences
def split_into_sentences(text):
    # Define separators as regex patterns
    separators = r"[■|•.\n]"

    # Split the text based on separators and filter non-empty sentences
    sentences = [sent.strip() for sent in re.split(separators, text) if sent.strip()]

    return sentences

# def train_embeddings(dataset, split="train"):
#     """Train word embeddings on the dataset."""
#     # Create corpus
#     corpus = []
#     for item in dataset[split]:
#         # Process article and summary
#         for text in [item['article'], item['summary']]:
#             sentences = split_into_sentences(text)
#             # sentences = split_into_chunks_with_llm(text)
#             for sent in sentences:
#                 words = re.findall(r'\b\w+\b', sent)
#                 if words:
#                     corpus.append(words)
#
#     # Train Word2Vec
#     model = Word2Vec(sentences=corpus,
#                      vector_size=100,
#                      window=5,
#                      min_count=2,
#                      workers=4)
#
#     return model.wv

# def create_matching_matrix(source_sentences, target_sentences, word_embeddings,
#                            neighbor_threshold=0.7, max_neighbors=5):
#     """
#     Create a matching matrix with similarity scores between sentences using enhanced TF-IDF.
#
#     Parameters:
#     - source_sentences (list of str): A list of sentences from the source text
#     - target_sentences (list of str): A list of sentences from the target text
#     - word_embeddings: Word embeddings model (e.g., Word2Vec, FastText)
#     - neighbor_threshold: Minimum similarity for semantic neighbors
#     - max_neighbors: Maximum number of neighbors per word
#
#     Returns:
#     - tuple: (source_sentences, target_sentences, matching_matrix, normalized_matrix)
#     """
#     # Initialize enhanced vectorizer
#     vectorizer = EnhancedTfidfVectorizer(
#         word_embeddings=word_embeddings,
#         neighbor_threshold=neighbor_threshold,
#         max_neighbors=max_neighbors,
#         analyzer='word',  # Changed to word-level analysis
#         token_pattern=r'\b\w+\b',  # Basic word tokenization
#         max_features=10000
#     )
#
#     # Combine source and target sentences for vectorization
#     all_sentences = source_sentences + target_sentences
#     tfidf_matrix = vectorizer.fit_transform(all_sentences)
#
#     # Separate TF-IDF representations
#     source_matrix = tfidf_matrix[:len(source_sentences)]
#     target_matrix = tfidf_matrix[len(source_sentences):]
#
#     # Compute cosine similarity matrix
#     matching_matrix = cosine_similarity(target_matrix, source_matrix)
#
#     # Normalize the similarity matrix
#     normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
#
#     return source_sentences, target_sentences, matching_matrix, normalized_matrix


# def create_matching_matrix(source_sentences, target_sentences):
#     """
#     Create a matching matrix with similarity scores between sentences in source_sentences and target_sentences.
#
#     Parameters:
#     - source_sentences (list of str): A list of sentences from the source text.
#     - target_sentences (list of str): A list of sentences from the target text.
#
#     Returns:
#     - np.ndarray: A matrix of similarity scores (target sentences × source sentences).
#     - np.ndarray: Normalized similarity matrix.
#     """
#     # Initialize TF-IDF Vectorizer (supports Hebrew using char-level ngrams)
#     vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
#
#     # Combine source and target sentences for vectorization
#     tfidf_matrix = vectorizer.fit_transform(source_sentences + target_sentences)
#
#     # Separate TF-IDF representations
#     source_matrix = tfidf_matrix[:len(source_sentences)]
#     target_matrix = tfidf_matrix[len(source_sentences):]
#
#     # Compute cosine similarity matrix
#     matching_matrix = cosine_similarity(target_matrix, source_matrix)
#
#     # Normalize the similarity matrix for easier interpretation
#     normalized_matrix = matching_matrix / matching_matrix.max(axis=1, keepdims=True)
#
#     return source_sentences, target_sentences, matching_matrix, normalized_matrix
