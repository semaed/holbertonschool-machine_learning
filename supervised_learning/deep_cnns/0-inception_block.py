#!/usr/bin/env python3
"""
Deep CNNs Module
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions(2014)
    Args:
        A_prev (ndarray): output from the previous layer
        filters (tuple or list): contains F1, F3R, F3, F5R, F5, FPP
    Returns:
        concatenated output of the inception block
    """
    # Ensure filters is a list or tuple of the correct length
    if not isinstance(filters, (list, tuple)) or len(filters) != 6:
        raise ValueError("filters must be a list or tuple of length 6")

    activation = 'relu'
    seed = 42  # Choose any integer as the seed for reproducibility
    # Provide seed to the initializer
    init = K.initializers.he_normal(seed=seed)
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                              activation=activation,
                              kernel_initializer=init)(A_prev)

    # 3x3 convolution, preceded by a 1x1 convolution to reduce dimensionality
    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                     activation=activation,
                                     kernel_initializer=init)(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              activation=activation,
                              kernel_initializer=init)(conv3x3_reduce)

    # 5x5 convolution, preceded by a 1x1 convolution to reduce dimensionality
    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                     activation=activation,
                                     kernel_initializer=init)(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                              activation=activation,
                              kernel_initializer=init)(conv5x5_reduce)

    # Pooling layer, followed by a 1x1 convolution to reduce dimensionality
    pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                 padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                                activation=activation,
                                kernel_initializer=init)(pool)

    # Concatenate the outputs of the 1x1 conv, 3x3 conv, 5x5 conv, and max pool layers
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, pool_proj])

    return output
