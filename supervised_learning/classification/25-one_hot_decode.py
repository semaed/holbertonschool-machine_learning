#!/usr/bin/env python3
"""
module one_hot_decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix into
    a vector of labels"""
    if type(one_hot) is np.ndarray and len(one_hot.shape) == 2:
        return np.argmax(one_hot, axis=0)
    return None
