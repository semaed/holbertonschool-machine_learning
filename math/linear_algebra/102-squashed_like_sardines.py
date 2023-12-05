#!/usr/bin/env python3
"""Module Concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""

    # Convert 1D lists to 2D lists
    if isinstance(mat1[0], int):
        mat1 = [[item] for item in mat1]
    if isinstance(mat2[0], int):
        mat2 = [[item] for item in mat2]

    # Check if mat1 and mat2 are lists of lists (2D matrices)
    if (not all(isinstance(row, list) for row in mat1) or
            not all(isinstance(row, list) for row in mat2)):
        return None

    # Check if the dimensions of the matrices match
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Recursive function to concatenate matrices along a specific axis
    def concat_recursive(m1, m2, ax):
        if ax == 0:
            return m1 + m2
        else:
            return [concat_recursive(m1[i], m2[i], ax-1) for i in range(len(m1))]

    # Call the recursive function
    return concat_recursive(mat1, mat2, axis)
