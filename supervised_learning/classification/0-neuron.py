#!/usr/bin/env python3
""""Neuron class defines single neuron performing binary classification"""


import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
        """Neuron class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive integer")

        self.b = 0
        self.W = np.random.normal(size=(1, nx))
        self.A = 0
