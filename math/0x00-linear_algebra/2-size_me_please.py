#!/usr/bin/env python3
"""
    Calculates the shape of a matrix.
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
