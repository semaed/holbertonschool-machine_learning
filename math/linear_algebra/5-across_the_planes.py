#!/usr/bin/env python3
"""
    Module Adds two matrices element-wise and Calculates the shape of a matrix
"""


def add_matrices2D(mat1, mat2):
    """
        Adds two matrices element-wise
            Args:
                mat1: first matrix
                mat2: second matrix
            Return:
                new matrix with sum of mat1 and mat2
    """


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
