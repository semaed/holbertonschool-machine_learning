#!/usr/bin/env python3
""" Neuron class that defines a single neuron performing
    binary classification
"""
import numpy as np


class Neuron:

    """
       Class constructor
    """

    def __init__(self, nx):
        """
            __init__: class constructor

            Input:
                @nx: is the number of input features to the neuron.

            Private Instances:
                @W: The weight vector for the neuron.
                @b: The bias for the neuron.
                @A: The activated output of the neuron (prediction).

            Raises:
                TypeError: nx must be an integer.
                ValueError: nx must be a positive integer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__b = 0
        self.__W = np.random.normal(size=(1, nx))
        self.__A = 0

    @property
    def W(self):
        """
            Getter function

            Return: The weight vector for the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
            Getter function

            Return: The bias for the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
            Getter function

            Return: The activated output of the neuron (prediction).
        """
        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron

            @X: is a numpy.ndarray with shape (nx, m) that contains
                the input data
                @nx: is the number of input features to the neurons
                @m: is the number of examples

            Return: the private attribute __A.
        """
        u = np.matmul(self.__W,  X) + self.__b
        self.__A = 1 / (1 + np.exp(-u))
        return self.__A

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression

            @Y: is a numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            @A: is a numpy.ndarray with shape (1, m) containing activated
                output of the neuron for each example

            Return: The cost
        """
        # I extract the first column where the data is located
        m = Y.shape[1]

        # Apply the formula
        # J(0) = -1/m Sigmoid[y(i) log(h0(x(i))) + (1 - y(i) log(1-h0(x(i)))]
        cost = (- 1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1-Y, np.log(1.0000001-A)))
        return cost
