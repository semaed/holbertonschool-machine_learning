#!/usr/bin/env python3
"""
Defines a function that calculates the accuracy of a prediction
for the neural network
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction for the neural network

    parameters:
        y [tf.placeholder]: placeholder for labels of the input data
        y_pred [tensor]: contains network's predictions

    returns:
        tensor containing decimal accuracy of the prediction
    """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
