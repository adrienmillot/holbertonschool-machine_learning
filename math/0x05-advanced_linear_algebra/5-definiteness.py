#!/usr/bin/env python3
"""
    Matrix Definiteness Calculation Module
"""

import numpy as np


def is_pos_def(matrix):
    """
        Check if the matrix is positive definite.

        Args:
            matrix (list): The given matrix.

        Returns:
            bool: True if the matrix is positive definite,
                  False otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_neg_def(matrix):
    """
        Check if the matrix is negative definite.

        Args:
            matrix (list): The given matrix.

        Returns:
            bool: True if the matrix is negative definite,
                  False otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) < 0)


def is_pos_semi_def(matrix):
    """
        Check if the matrix is positive semi-definite.

        Args:
            matrix (list): The given matrix.

        Returns:
            bool: True if the matrix is positive semi-definite,
                  False otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) >= 0)


def is_neg_semi_def(matrix):
    """
        Check if the matrix is negative semi-definite.

        Args:
            matrix (list): The given matrix.

        Returns:
            bool: True if the matrix is negative semi-definite,
                  False otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) <= 0)


def check_symmetric_matrix(matrix: np.ndarray):
    """
        Check if the matrix is symmetric.

        Args:
            matrix (list): The given matrix.

        Returns:
            void: Raise only error if not symmetric.
    """
    row, column = matrix.shape

    if row != column:
        raise TypeError

    matrix_transpose = matrix.copy().T

    if not np.array_equal(matrix, matrix_transpose):
        raise TypeError


def definiteness(matrix):
    """
        Determines the definiteness of the given matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            list: The definiteness.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    try:
        check_symmetric_matrix(matrix)

    except Exception as exception:
        return None

    if is_pos_def(matrix):
        msg = "Positive definite"
    elif is_neg_def(matrix):
        msg = "Negative definite"
    elif is_pos_semi_def(matrix):
        msg = "Positive semi-definite"
    elif is_neg_semi_def(matrix):
        msg = "Negative semi-definite"
    else:
        msg = "Indefinite"

    return msg
