#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the Adam optimization algorithm
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    """
    op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return op.minimize(loss)
