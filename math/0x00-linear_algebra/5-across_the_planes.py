#!/usr/bin/env python3
"""
    Adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
        Returns the sum of two matrices
    """

    if len(mat1) == 0 or len(mat2) == 0:
        return None

    if len(mat1[0]) != len(mat2[0]):
        return None

    result = []
    for row in range(len(mat1)):
        result.append([])
        for column in range(len(mat1[0])):
            result[row].append(mat1[row][column] + mat2[row][column])

    return result
