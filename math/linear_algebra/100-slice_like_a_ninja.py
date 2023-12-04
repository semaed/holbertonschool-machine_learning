#!/usr/bin/env python3
"""
function def np_slice(matrix, axes={})
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes
    """
    slices = [slice(None)] * \
        matrix.ndim

    for axis, slice_params in axes.items():
        slices[axis] = slice(*slice_params)

    return matrix[tuple(slices)]
