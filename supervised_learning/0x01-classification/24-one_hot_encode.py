#!/usr/bin/env python3
"""
    One-hot encoding.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        Convert a class vector (integers) to a one-hot matrix.

        Args:
            Y (np.ndarray): class vector.
            classes (int): number of classes.

        Returns:
            np.ndarray: one-hot matrix.
    """

    if type(Y) is not np.ndarray:
        return None

    if type(classes) is not int:
        return None

    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
