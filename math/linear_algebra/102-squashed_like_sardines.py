#!/usr/bin/env python3
"""Module Concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""

    # Convert 1D lists to 2D lists
    if isinstance(mat1[0], int):
        mat1 = [[item] for item in mat1]
    if isinstance(mat2[0], int):
        mat2 = [[item] for item in mat2]

    # Create deep copies of the input matrices
    mat1 = [row[:] for row in mat1]
    mat2 = [row[:] for row in mat2]

    # Check if mat1 and mat2 are lists of lists (2D matrices)
    if (not all(isinstance(row, list) for row in mat1) or
            not all(isinstance(row, list) for row in mat2)):
        return None

    # Concatenate along axis 0
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Concatenate along axis 1
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    # If axis is neither 0 nor 1, return None
    else:
        return None
