#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Function that creates a layer of a neural network using dropout
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
