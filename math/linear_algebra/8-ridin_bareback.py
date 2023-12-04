#!/usr/bin/env python3
"""
    Module Matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
        Performs matrix multiplication
            Args:
                mat1: first matrix
                mat2: second matrix
            Return:
                new matrix with mat1 * mat2
    """
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = []
    for i in range(len(mat1)):
        new_mat.append([])
        for j in range(len(mat2[0])):
            new_mat[i].append(0)
            for k in range(len(mat2)):
                new_mat[i][j] += mat1[i][k] * mat2[k][j]
    return new_mat
