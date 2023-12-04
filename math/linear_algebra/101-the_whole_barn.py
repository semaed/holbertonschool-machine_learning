#!/usr/bin/env python3
"""
Add two matrices
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices
        Args:
            mat1: first matrix
            mat2: second matrix
        Return:
            new matrix with mat1 and mat2 added
    """

    if not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2

    if isinstance(mat1, list) and isinstance(mat2, list) and not isinstance(mat1[0], list) and not isinstance(mat2[0], list):
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if not isinstance(mat1[i], list) or not isinstance(mat2[i], list) or len(mat1[i]) != len(mat2[i]):
            return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
