#!/usr/bin/env python3
"""
    Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
        Returns the sum of two arrays
    """
    result = arr1.copy()

    if len(arr1) != len(arr2):
        return None

    for row in range(len(arr1)):
        result[row] = arr1[row] + arr2[row]

    return result
