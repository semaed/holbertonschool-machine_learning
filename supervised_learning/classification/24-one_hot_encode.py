#!/usr/bin/env python3
"""
    Module documentation
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector
    into a one-hot matrix """
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception as e:
        return None
