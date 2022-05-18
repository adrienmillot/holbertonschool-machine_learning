#!/usr/bin/env python3
"""
    Shuffles the data
"""

import numpy as np


def shuffle_data(X, Y):
    """
        Shuffles the data

        Args:
            X: numpy.ndarray of shape (m, nx) to shuffle
            Y: numpy.ndarray of shape (m, ny) to shuffle

        Returns:
            X: shuffled numpy.ndarray of shape (m, nx)
            Y: shuffled numpy.ndarray of shape (m, ny)
    """

    permutation = np.random.permutation(Y.shape[0])
    X = X[permutation, :]
    Y = Y[permutation]

    return X, Y
