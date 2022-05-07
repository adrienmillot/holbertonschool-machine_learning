#!/usr/bin/env python3
"""
    One-hot decoding.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
        Convert a one-hot matrix to a class vector.

        Args:
            one_hot (np.ndarray): one-hot matrix.

        Returns:
            np.ndarray: class vector.
    """

    if type(one_hot) is not np.ndarray:
        return None

    if len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
