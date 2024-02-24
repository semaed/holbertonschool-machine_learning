#!/usr/bin/env python3
"""LeNet-5 implementation in TensorFlow 1.x"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using TensorFlow 1.x
    Args:
    x: tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
    y: tf.placeholder of shape (m, 10) containing the one-hot labels for the network

    Returns:
    - A tensor for the softmax activated output
    - A training operation that utilizes Adam optimization
    - A tensor for the loss of the network
    - A tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First Convolutional Layer
    conv1 = tf.layers.conv2d(x, 6, 5, padding='same',
                             activation=tf.nn.relu, kernel_initializer=init)
    # Pooling Layer
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    # Second Convolutional Layer
    conv2 = tf.layers.conv2d(
        pool1, 16, 5, padding='valid', activation=tf.nn.relu, kernel_initializer=init)
    # Pooling Layer
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    # Flattening the output
    flat = tf.layers.flatten(pool2)

    # Fully Connected Layer 1
    fc1 = tf.layers.dense(flat, 120, activation=tf.nn.relu,
                          kernel_initializer=init)
    # Fully Connected Layer 2
    fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.relu,
                          kernel_initializer=init)
    # Softmax Output Layer
    output = tf.layers.dense(
        fc2, 10, activation=tf.nn.softmax, kernel_initializer=init)

    # Loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return output, optimizer, loss, accuracy
