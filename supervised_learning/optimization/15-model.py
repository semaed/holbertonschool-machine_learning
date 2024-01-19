#!/usr/bin/env python3
"""
Defines function that builds, trains, and saves a neural network model
using TensorFlow using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization
"""
import tensorflow.compat.v1 as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way:
    """
    permutation = np.random.permutation(len(X))

    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y


def create_placeholders(nx, classes):
    """
    Function to create the placeholders
    """
    x = tf.placeholder('float', [None, nx], name='x')
    y = tf.placeholder('float', [None, classes], name='y')

    return x, y


def create_layer(prev, n, activation):
    """
    Function that creates the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')

    return layer(prev)


def forward_prop(x, layer_sizes, activations, epsilon=1e-8):
    """
    Function that creates the forward propagation graph for the NN
    """
    for i in range(len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            layer = create_batch_norm_layer(x, layer_sizes[i], activations[i],
                                            epsilon=1e-8)
        else:
            layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return layer


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuranvy of a prediction.
    """
    eq = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))

    return accuracy


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross.entropy loss of a prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon=1e-8):
    """
    Function that creates the training operation for a NN in tensorflow
    using the Adam optimization algorithm
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    optimized_adam = optimizer.minimize(loss)

    return optimized_adam


def create_batch_norm_layer(prev, n, activation, epsilon=1e-8):
    """
    Function that creates a batch normalization layer for a NN in tensorflow:
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)

    z = layer(prev)

    mt, vt = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    zt = tf.nn.batch_normalization(z, mt, vt, beta, gamma, epsilon)
    y_pred = activation(zt)

    return y_pred


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    """

    decay = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)

    return decay


def get_batch(t, batch_size):
    """
    Helper function to divide data in batches
    """

    batch_list = []
    i = 0
    m = t.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(t[i:(i + batch_size)])
        else:
            batch_list.append(t[i:])
        i += batch_size

    return batch_list


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a NN model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization:
    """

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    m = Data_train[0].shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign_add(global_step, 1,
                                          name='increment_global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs + 1):
            x_t, y_t = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run((loss, accuracy),
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if e < epochs:
                X_batch_t = get_batch(x_t, batch_size)
                Y_batch_t = get_batch(y_t, batch_size)
                for b in range(1, len(X_batch_t) + 1):
                    sess.run((increment_global_step, train_op),
                             feed_dict={x: X_batch_t[b - 1],
                             y: Y_batch_t[b - 1]})
                    loss_t, acc_t = sess.run((loss, accuracy),
                                             feed_dict={x: X_batch_t[b - 1],
                                                        y: Y_batch_t[b - 1]})
                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_t))
                        print('\t\tAccuracy: {}'.format(acc_t))
        save_path = saver.save(sess, save_path)
    return save_path
