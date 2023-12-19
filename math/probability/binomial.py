#!/usr/bin/env python3
""" defines Binomial class that represents binomial distribution """


class Binomial:
    """
    class that represents Binomial distribution

    class constructor:
        def __init__(self, data=None, n=1, p=0.5)

    instance attributes:
        n [int]: the number of Bernoilli trials
        p [float]: the probability of a success

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            n [int]: the number of Bernoilli trials
            p [float]: the probability of a success

        Sets the instance attributes n and p
        If data is not given:
            Use the given n and p
            Raise ValueError if n is not positive value
            Raise ValueError if p is not a valid probability
        If data is given:
            Calculate n and p from data, rounding n to nearest int
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(i - mean)**2 for i in data]) / len(data)
            self.n = round(mean**2 / (mean - variance))
            self.p = mean / self.n

    def pmf(self, k):
        """
        calculates PMF for given number of successes

        parameters:
            k [int]: the number of successes

        returns:
            PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        p = self.p
        n = self.n
        q = (1 - p)
        n_factorial = 1
        for i in range(n):
            n_factorial *= (i + 1)
        k_factorial = 1
        for i in range(k):
            k_factorial *= (i + 1)
        nk_factorial = 1
        for i in range(n - k):
            nk_factorial *= (i + 1)
        binomial_co = n_factorial / (k_factorial * nk_factorial)
        pmf = binomial_co * (p ** k) * (q ** (n - k))
        return pmf
