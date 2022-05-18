#!/usr/bin/env python3
"""
    Normalize a dataset X
"""


def normalize(X, mean, standard_deviation):
    """
        Normalizes a dataset X

        Args:
            X: numpy.ndarray of shape (m, nx) to be normalized
            mean: mean of all features
            standard_deviation: standard deviation of all features

        Returns:
            X: normalized dataset
    """

    X = (X - mean) / standard_deviation

    return X
