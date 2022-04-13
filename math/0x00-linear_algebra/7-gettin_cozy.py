#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Returns the concatenation of two matrices
    """

    # Vertical
    if (axis == 0):
        if len(mat1[0]) != len(mat2[0]):
            return None

        result = [row[:] for row in mat1]   # Copy of the matrix 1
        result.extend(mat2)                 # Add the matrix 2

    # Horizontal
    elif (axis == 1):
        if len(mat1) != len(mat2):
            return None

        result = []
        for row in range(len(mat2)):
            result.append(mat1[row] + mat2[row])

    return result
