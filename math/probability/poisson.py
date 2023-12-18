#!/usr/bin/env python3
""" defines Poisson class that represents Poisson distribution """


class Poisson:
    """
    class that represents Poisson distribution

    class constructor:
        def __init__(self, data=None, lambtha=1.)

    instance attributes:
        lambtha [float]: the expected number of occurances in a given time

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distribution
            lambtha [float]: the expected number of occurances in a given time
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """calculates PMF for given number of successes"""

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return ((2.7182818285 ** (-self.lambtha)) *
                (self.lambtha ** k)) / factorial
