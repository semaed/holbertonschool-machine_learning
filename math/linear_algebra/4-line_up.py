#!/usr/bin/env python3
"""
    Module content:
        - add_array: Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
        Adds two arrays element-wise
            Args:
                arr1: first array
                arr2: second array
            Return:
                new array with sum of arr1 and arr2
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
