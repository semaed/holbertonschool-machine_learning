#!/usr/bin/env python3
"""
module create_momentum_op
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm
    """
    op = tf.train.MomentumOptimizer(alpha, beta1)
    return op.minimize(loss)
