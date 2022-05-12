#!/usr/bin/env python3
"""
    Creates the training operation
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
        Creates the training operation

        Args:
            loss: loss tensor
            alpha: learning rate

        Returns:
            train_op: training operation
    """

    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return train_op
