#!/usr/bin/env python3
"""
    Calculates the specificity for each class
"""

import numpy as np


def specificity(confusion):
    """
        Calculates the specificity for each class

        Args:
            confusion: the confusion matrix

        Returns:
            a numpy.ndarray of specificity for each class
    """

    true_positives = np.diagonal(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - false_positives - \
        false_negatives - true_positives

    return true_negatives / (true_negatives + false_positives)
