#!/usr/bin/env python3
"""Script to create an inception network"""
import tensorflow.keras as K
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Assuming '0-inception_block.py' is correctly implemented
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function to create an inception network.
    """
    X = K.Input(shape=(224, 224, 3))

    activation = 'relu'

    # First convolutional layer
    conv_1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2),
                             padding='same', activation=activation,
                             kernel_initializer=K.initializers.HeNormal())(X)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(conv_1)
    # Second convolutional layer
    conv_2P = K.layers.Conv2D(filters=64, kernel_size=1, padding='valid',
                              activation=activation,
                              kernel_initializer=K.initializers.HeNormal())(max_pool_1)
    conv_2 = K.layers.Conv2D(filters=192, kernel_size=3, padding='same',
                             activation=activation,
                             kernel_initializer=K.initializers.HeNormal())(conv_2P)
    max_pool_2 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(conv_2)

    # Inception blocks
    iblock_1 = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    iblock_2 = inception_block(iblock_1, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(iblock_2)

    iblock_3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    iblock_4 = inception_block(iblock_3, [160, 112, 224, 24, 64, 64])
    iblock_5 = inception_block(iblock_4, [128, 128, 256, 24, 64, 64])
    iblock_6 = inception_block(iblock_5, [112, 144, 288, 32, 64, 64])
    iblock_7 = inception_block(iblock_6, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(iblock_7)

    iblock_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    iblock_9 = inception_block(iblock_8, [384, 192, 384, 48, 128, 128])

    # Final average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=(1, 1),
                                         padding='valid')(iblock_9)

    # Dropout layer
    dropout = K.layers.Dropout(0.4)(avg_pool)

    # Final dense layer
    FC = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=K.initializers.HeNormal())(dropout)

    model = K.models.Model(inputs=X, outputs=FC)

    return model


# If you have a main block to instantiate and test the model, you can do so like this:
if __name__ == "__main__":
    model = inception_network()
    model.summary()
