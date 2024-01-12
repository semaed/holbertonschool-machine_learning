#!/usr/bin/env python3
"""
module calculate_accuracy
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
