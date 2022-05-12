#!/usr/bin/env python3
"""
    Creates the placeholders needed for the model
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
        Creates the placeholders needed for the model

        Args:
            nx: number of input features
            classes: number of classes

        Returns:
            x: placeholder for the input data
            y: placeholder for the input labels
    """

    x = tf.placeholder(tf.float32, [None, nx], name='x')
    y = tf.placeholder(tf.float32, [None, classes], name='y')

    return x, y
