#!/usr/bin/env python3
"""Neural network with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
        NeuralNetwork with one hidden layer performing binary
        classification
    """

    def __init__(self, nx, nodes):
        """
            __init__: class constructor

            Input:
                @nx: the number of input features
                @nodes: the number of nodes found in the hidden layer

            Raises:
                TypeError:
                ValueError:
                TypeError:
                ValueError:
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be a integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
