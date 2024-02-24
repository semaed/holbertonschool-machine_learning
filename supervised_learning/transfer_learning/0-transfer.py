#!/usr/bin/env python3
"""Transfer learning CIFAR-10 in densenet 121"""

import keras
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Scale up the training and testing data to 224x224 pixels
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create a lambda layer to scale up the data
lambda_layer = keras.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))

# Load the pre-trained Keras application
application = keras.applications.resnet50

# Freeze most of the application layers
for layer in application.layers[:-5]:
    layer.trainable = False

# Create the model
model = keras.Sequential([
    lambda_layer,
    application,
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the trained model
model.save('cifar10.h5')


def preprocess_data(X, Y):
    """ pre-processes the data for your model """
    # Scale up the training and testing data to 224x224 pixels
    x_train = X.astype('float32') / 255.0
    x_test = Y.astype('float32') / 255.0

    # Create a lambda layer to scale up the data
    lambda_layer = keras.layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224)))

    # Load the pre-trained Keras application
    application = keras.applications.resnet50

    # Freeze most of the application layers
    for layer in application.layers[:-5]:
        layer.trainable = False

    # Create the model
    model = keras.Sequential([
        lambda_layer,
        application,
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Save the trained model
    model.save('cifar10.h5')

    return x_train, x_test
