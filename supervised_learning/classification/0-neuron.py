#!/usr/bin/env python3
""""Neuron class defines single neuron performing binary classification"""


import numpy as np


class Neuron:
    def __init__(self, nx):
        """_summary_

        Args:
            nx (_type_): _description_
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
