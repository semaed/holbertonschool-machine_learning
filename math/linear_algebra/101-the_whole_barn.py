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

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return mat1 + mat2 if not isinstance(mat1, list) \
            and not isinstance(mat2, list) else None

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        temp = add_matrices(mat1[i], mat2[i])
        if temp is None:
            return None
        result.append(temp)

    return result
