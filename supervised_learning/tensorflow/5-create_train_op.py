#!/usr/bin/env python3
"""
module create_train_op
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
