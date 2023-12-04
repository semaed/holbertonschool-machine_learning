#!/usr/bin/env python3
"""
    Module Performs element-wise addition,subtraction, multiplication,
      and division.
"""


def np_elementwise(mat1, mat2):
    """
        np_elementwise - Performs element-wise addition,
                        subtraction, multiplication, and
                        division.

        @mat1 & @mat2: Numpy arrays to perform operations on

        Return: Tuple containing numpy.ndarrays with the
                result of the operations.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
