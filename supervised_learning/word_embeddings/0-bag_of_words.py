#!/usr/bin/env python3
""" Module Word Embeddings """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
      Create a bag of words embedding matrix
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    if vocab is None:
        vocab = []
    embedding = vectorizer.fit_transform(sentences).toarray()
    vocab = list(vectorizer.get_feature_names())
    return embedding, vocab
