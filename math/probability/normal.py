#!/usr/bin/env python3
"""Create a class Normal that represents a normal distribution"""


class Normal:
    """class that represents normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            mean [float]: the mean of the distribution
            stddev [float]: the standard deviation of the distribution

        Sets the instance attributes mean and stddev as floats
        If data is not given:
            Use the given mean and stddev
            raise ValueError if stddev is not positive value
        If data is given:
            Calculate the mean and stddev of data
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(i - self.mean)**2 for i in data]) /
                           len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        parameters:
            x [float]: the x-value

        returns:
            the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        parameters:
            z [float]: the z-score

        returns:
            the x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        parameters:
            x [float]: the x-value

        returns:
            the PDF value for x
        """
        return (2.7182818285 ** (-0.5 * ((x - self.mean) / self.stddev) ** 2) /
                (self.stddev * (2 * 3.1415926536) ** 0.5))
