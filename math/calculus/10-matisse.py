#!/usr/bin/env python3
"""
    Module Calculate the derivative of a polynomial
"""


def poly_derivative(poly):
    """
        Calculates the derivative of a polynomial
        poly: list of coefficients representing a polynomial
        Return: a new list of coefficients representing the
        derivative of the polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(1, len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        derivative.append(i * poly[i])
    return derivative
