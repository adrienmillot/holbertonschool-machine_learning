#!/usr/bin/env python3
"""
    Creates a layer of a neural network
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
        Creates a layer of a neural network

        Args:
            prev: tensorflow tensor object of the layer previous to this layer
            n: number of nodes in the layer
            activation: activation function

        Returns:
            tensorflow tensor object of the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )

    return layer(prev)
