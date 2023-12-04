#!/usr/bin/env python3
"""
    Module Returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
        Returns the transpose of a 2D matrix
            Args:
                matrix: matrix to transpose
            Return:
                transpose of matrix
    """
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
    for row in matrix:
        for i in range(len(row)):
            transpose[i].append(row[i])
    return transpose
