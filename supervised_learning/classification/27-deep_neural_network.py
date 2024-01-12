#!/usr/bin/env python3
"""
    Module Content:
        - Defines a deep neural network performing binary
            classification
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
        Defines a deep neural network performing
        binary classification
    """

    def __init__(self, nx, layers):
        """
            Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" +
                           str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """
            getter function for L
        """
        return self.__L

    @property
    def cache(self):
        """
            getter function for cache
        """
        return self.__cache

    @property
    def weights(self):
        """
            getter function for weights
        """
        return self.__weights

    def train(self, X_train, Y_train, iterations=1000, learning_rate=0.5):
        """
        Trains the neural network
        """
        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X_train)

            # Calculate cost
            cost = self.cost(Y_train, A)

            # Backward propagation
            dZ, dW, db = self.backward_prop(A, Y_train)

            # Update weights and biases
            self.update_weights(dW, db, learning_rate)

        return self

    @classmethod
    def load(cls, file_path):
        """
        Loads a neural network from a pickle file
        """
        with open(file_path, 'rb') as file:
            network = pickle.load(file)
        return network
