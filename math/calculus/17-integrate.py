#!/usr/bin/env python3
"""
    Module Calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
        Calculates the integral of a polynomial
        poly: list of coefficients representing a polynomial
        C: is an integer representing the integration constant
        Return: a new list of coefficients representing the
        integral of the polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int:
        return None
    if len(poly) == 1:
        if poly[0] == 0:
            return [C]
        else:
            return [C, poly[0]]
    integral = [C]
    for i in range(len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        coefficient = poly[i] / (i + 1)
        if coefficient.is_integer():
            coefficient = int(coefficient)
        integral.append(coefficient)
    return integral
