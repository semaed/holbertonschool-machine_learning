#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification
"""
import numpy as np


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
        for layer in range(self.__L):
            if type(layers[layer]) is not int or layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))
            if layer == 0:
                He_et_al = np.random.randn(layers[layer], nx) * np.sqrt(2/nx)
                self.__weights['W' + str(layer + 1)] = He_et_al
            else:
                He_et_al = np.random.randn(layers[layer], layers[layer - 1]) *\
                    np.sqrt(2/layers[layer-1])
                self.__weights['W' + str(layer + 1)] = He_et_al

    @property
    def L(self):
        """ Getter function """
        return self.__L

    @property
    def cache(self):
        """ Getter function """
        return self.__cache

    @property
    def weights(self):
        """ Getter function """
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation"""
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            fwp_A = np.matmul(weights["W" + str(layer + 1)], cache["A" +
                              str(layer)])
            fwp = fwp_A + weights["b" + str(layer + 1)]
            cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-fwp))
        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1-Y, np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """
            Evaluate the neuron's predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate Gradient descent """
        m = Y.shape[1]
        weights = self.__weights.copy()
        for layer in reversed(range(self.__L)):
            A = cache["A" + str(layer + 1)]
            if layer == self.__L - 1:
                dZ = A - Y
            else:
                dZ = np.matmul(weights["W" + str(layer + 2)].T, dZ) * (A *
                                                                       (1-A))
            dW = np.matmul(dZ, cache["A" + str(layer)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(layer + 1)] = weights["W" +
                                                           str(layer + 1)] -\
                alpha * dW
            self.__weights["b" + str(layer + 1)] = weights["b" +
                                                           str(layer + 1)] -\
                alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train the neural network"""
        if type(iterations) != int:
            raise TypeError("interations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for iteration in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)
