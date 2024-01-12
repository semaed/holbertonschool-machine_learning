#!/usr/bin/env python3
"""
Defines a function to create a layer for neural network
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Creates a layer for neural network

    parameters:
        prev [tensor]: tensor output of the previous layer
        n [int]: the number of nodes in the layer to create
        activation [function]: the activation function the layer should use

    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        to implement He et. al initialization for the layer weights
    each layer is given the name "layer"

    returns:
        tensor output of the layer
    """
    weight = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=weight, name='layer')
    return layer(prev)
