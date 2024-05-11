#!/usr/bin/env python3
"""Module bag_of_words."""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.

    Args:
        sentences (list): sentences to analyze
        vocab (list, optional): vocabulary words to use for the analysis.
                                Defaults to None.
    Returns:
        embeddings: contains the embeddings
        features: list of the features used for embeddings
    """
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names_out()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embedding = X.toarray()
    return embedding, vocab
