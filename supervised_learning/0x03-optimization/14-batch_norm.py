#!/usr/bin/env python3
"""
    Creates a batch normalization layer for a neural network in tensorflow
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Creates a batch normalization layer for a neural network in tensorflow

        Args:
            prev: a tensor of shape (batch, n_prev)
            n: size of the layer
            activation: activation function of the layer

        Returns:
            a tensor of shape (batch, n)
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)

    # Calculate the mean and variance of layer
    # for simple batch normalization pass axes=[0] (batch only).
    mean, variance = tf.nn.moments(layer(prev), axes=[0])

    # gamma and beta, initialized as vectors of 1 and 0 respectively
    gamma = tf.ones([n])
    beta = tf.zeros([n])

    epsilon = 1e-8
    batch_normalization_output = tf.nn.batch_normalization(
        x=layer(prev), mean=mean,
        variance=variance, offset=beta,
        scale=gamma, variance_epsilon=epsilon)

    return activation(batch_normalization_output)
