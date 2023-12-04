#!/usr/bin/env python3
"""
Module Calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix
        Args:
            matrix: matrix to calculate shape of
        Return:
            shape: list of integers representing shape of matrix
    """
    shape = []
    if type(matrix) is list:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    return shape
