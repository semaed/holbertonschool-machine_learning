#!/usr/bin/env python3
"""
Defines a function that creates the training operation
for the neural network
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network

    parameters:
        loss [tensor]: loss of the network's prediction
        alpha [float]: learning rate

    returns:
        operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
