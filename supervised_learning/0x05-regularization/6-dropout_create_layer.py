#!/usr/bin/env python3
"""
    Creates a layer of the neural network using dropout
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a layer of the neural network using dropout

        Args:
            prev: tensor containing the output of the previous layer
            n: number of nodes in the layer to create
            activation: activation function to use
            keep_prob: probability that a node will be kept

        Returns:
            A: numpy.ndarray of shape (n, m) containing the output of the layer
            cache: dictionary containing the inputs and outputs of the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    drop_out = tf.layers.Dropout(rate=1 - keep_prob)
    layer = tf.layers.Dense(
        units=n, kernel_initializer=initializer, activation=activation)

    return drop_out(layer(prev))
