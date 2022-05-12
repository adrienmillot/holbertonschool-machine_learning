#!/usr/bin/env python3
"""
    Calculates the accuracy of a prediction
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction

        Args:
            y: real labels
            y_pred: predicted labels

        Returns:
            accuracy: accuracy of the prediction
    """

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
