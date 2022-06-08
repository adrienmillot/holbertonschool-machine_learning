#!/usr/bin/env python3
"""
    Adam Optimization
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
        Creates the operation to perform the Adam optimization

        Args:
            loss: loss of the network
            alpha: learning rate
            beta1: first Adam weight
            beta2: second Adam weight
            epsilon: small number to avoid division by zero in Adam

        Returns:
            train_op: the Adam optimization operation
    """

    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)
