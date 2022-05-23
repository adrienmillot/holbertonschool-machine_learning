#!/usr/bin/env python3
"""
    Calculates the sensitivity for each class
"""

import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity for each class

        Args:
            confusion: the confusion matrix

        Returns:
            a numpy.ndarray of sensitivity for each class
    """

    return np.diagonal(confusion) / np.sum(confusion, axis=1)
