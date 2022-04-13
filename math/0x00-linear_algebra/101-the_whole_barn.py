#!/usr/bin/env python3
"""
    Adds two matrices.
"""


def matrix_shape(matrix):
    """
        Returns the shape of a matrix
    """

    shape = []

    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape


def add_matrices(mat1, mat2):
    """
        Adds two matrices
    """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    result = []

    for row in range(len(mat1)):
        result.append([])
        if type(mat1[row]) is list:
            for col in range(len(mat1[row])):
                if type(mat1[row][col]) is list:
                    result[row].append(
                        add_matrices(mat1[row][col], mat2[row][col])
                    )
                else:
                    result[row].append(mat1[row][col] + mat2[row][col])
        else:
            result[row] = mat1[row] + mat2[row]

    return result
