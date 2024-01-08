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

    def evaluate(self, X, Y):
        """
            Evaluate the neuron's predictions

            @X: is a np.ndarray with shape(nx, m) the input data
                @nx: is the number of input features
                @m: is the number of examples
            @Y: is a np.ndarray with shape(1, m) the correct label
                @m: is the number of examples

            Return: The neuron's prediction and the cost of the network

            The prediction should be a numpy.ndarray with shape(1, m)
            containing the predicted labels for each example. The
            label values should be 1 if the output of the network
            is >= 0.5 and 0 otherwise.
        """
        self.forward_prop(X)

        cost = self.cost(Y, self.__A)
        pred = np.where(self.__A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron

            @X is a numpy.ndarray with shape(nx, m) the input data
                @nx: is the number of input features to the neuron
                @m: is the number of examples
             @Y: is a numpy.ndarray with shape (1, m) the correct label
                for the input data
                @m: is the number of examples
            @A: is a numpy.ndarray with shape (1, m) the activated output
                of the neuron for each example
            @alpha: is the learning rate
        """
        # I extract the first column where the data is located
        m = Y.shape[1]

        # Calculate the gradient of the cost function with respect to A.
        dA = A - Y

        # Apply the formula θnew = 0 old - a∇J(0old)
        dW = (1/m) * np.dot(dA, X.T)
        db = (1/m) * np.sum(dA)

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the neuron

            @X is a numpy.ndarray with shape(nx, m) the input data
                @nx: is the number of input features to the neuron
                @m: is the number of examples
            @Y: is a numpy.ndarray with shape (1, m) the correct label
                for the input data
                @m: is the number of examples
            @iterations: is the number of iteration to train over
            @alpha: is the learning rate

            Return: the evaluation of the training data after iterations
                of training have occurred.

            Raises:
                TypeError: iterations must be an integer
                ValueError: iterations must be a positive integer
                TypeError: alpha must be a float
                ValueError: alpha must be positive
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")
        for train in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
