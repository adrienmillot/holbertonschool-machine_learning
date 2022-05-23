#!/usr/bin/env python3
"""
    Calculates the f1 score for each class
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        Calculates the f1 score for each class

        Args:
            confusion: the confusion matrix

        Returns:
            a numpy.ndarray of f1 score for each class
    """

    prec = precision(confusion)
    sens = sensitivity(confusion)

    return 2 * (prec * sens) / (prec + sens)
