#!/usr/bin/env python3
"""
    Module content:
        - add_matrices2D - Adds two matrices element-wise
        - matrix_shape - Calculates the shape of a matrix
"""


def add_matrices2D(mat1, mat2):
    """
        add_matrices2D - Adds two matrices element-wise

        @mat1 & @mat2: Matrices to be added

        Return: Upon success returns a new matrix with the sum,
                otherwise None

        Description: In this function, there are two ways that will
        render the same result.
    """
    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
