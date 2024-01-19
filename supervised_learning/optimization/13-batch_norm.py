#!/usr/bin/env python3
"""
module batch_norm
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using batch
    normalization
    """
    m = Z.shape[0]
    mean = 1 / m * np.sum(Z, axis=0)
    variance = 1 / m * np.sum((Z - mean) ** 2, axis=0)
    Z_norm = (Z - mean) / (np.sqrt(variance + epsilon))
    return (gamma * Z_norm) + beta
