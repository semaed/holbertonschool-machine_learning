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

    if len(mat1) != len(mat2):
        return None

    if isinstance(mat1[0], list):
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]

    return [mat1[i] + mat2[i] for i in range(len(mat1))]
