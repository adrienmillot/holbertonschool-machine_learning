#!/usr/bin/env python3
"""
    Calculates the precision for each class
"""

import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class

        Args:
            confusion: the confusion matrix

        Returns:
            a numpy.ndarray of precision for each class
    """

    return np.diagonal(confusion) / np.sum(confusion, axis=0)
