#!/usr/bin/env python3
"""
module create_layer
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    function that creates layer
    """
    weight = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=weight, name='layer')
    return layer(prev)
