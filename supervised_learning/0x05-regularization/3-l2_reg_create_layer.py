#!/usr/bin/env python3
"""
    Creates a layer of the neural network using L2 regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a layer of the neural network using L2 regularization

        Args:
            prev: the output of the previous layer
            n: the number of nodes the new layer should contain
            activation: the activation function that should be used on
                        the layer
            lambtha: L2 regularization parameter

        Returns:
            the activated output of the new layer
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
