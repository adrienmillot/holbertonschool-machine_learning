#!/usr/bin/env python3
"""
    Calculates the loss of a prediction
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
        Calculates the loss of a prediction

        Args:
            y: real labels
            y_pred: predicted labels

        Returns:
            loss: loss of the prediction
    """

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
