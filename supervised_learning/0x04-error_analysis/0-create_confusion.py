#!/usr/bin/env python3
"""
    Create a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Creates a confusion matrix

        Args:
            labels: one-hot containing the correct labels for each data point
            logits: one-hot containing the predicted labels

        Returns:
            a confusion matrix
    """

    confusion = np.zeros((labels.shape[1], logits.shape[1]))

    for i in range(labels.shape[0]):
        confusion[np.argmax(labels[i])][np.argmax(logits[i])] += 1

    return confusion
