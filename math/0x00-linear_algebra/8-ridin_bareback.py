#!/usr/bin/env python3
"""
    Performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
        Returns the multiplication of two matrices
    """

    if (
        # Different column size (1) and row size (2)
        len(mat2) != len(mat1[0])
    ):
        return None

    result = []

    for row in range(len(mat1)):
        result.append([])

        for column in range(len(mat2[0])):
            # Initialize the result matrix
            result[row].append(0)

            for i in range(len(mat2)):
                # multiply the row of mat1 with the column of mat2
                result[row][column] += mat1[row][i] * mat2[i][column]

    return result
