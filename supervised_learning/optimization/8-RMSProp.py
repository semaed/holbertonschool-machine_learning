#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the RMSProp optimization algorithm
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    """
    op = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                   epsilon=epsilon)
    step_op = op.minimize(loss)
    return step_op
