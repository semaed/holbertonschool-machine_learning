#!/usr/bin/env python3
""" Neuron class that defines a single neuron performing 
binary classification
"""
import numpy as np


class Neuron:

    """
        Function:
            - def __init__(self, nx): Class constructor
    """

    def __init__(self, nx):
        """
            __init__: class constructor

            Input:
                @nx: is the number of input features to the neuron.

            Public Instances:
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
            raise ValueError("nx must be positive integer")

        self.b = 0
        self.W = np.random.normal(size=(1, nx))
        self.A = 0

        @property
        def b(self):
            """
                b: getter function for b attribute
            """
            return self.__b

        @property
        def A(self):
            """
                A: getter function for A attribute
            """
            return self.__A

        @property
        def W(self):
            """
                W: getter function for W attribute
            """
            return self.__W
