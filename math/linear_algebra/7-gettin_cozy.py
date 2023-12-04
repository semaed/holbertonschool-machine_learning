#!/usr/bin/env python3
"""
    Module Concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Concatenates two matrices along a specific axis
            Args:
                mat1: first matrix
                mat2: second matrix
                axis: axis to concatenate
            Return:
                new matrix with mat1 and mat2 concatenated
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    else:
        return None
