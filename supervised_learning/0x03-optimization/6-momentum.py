#!/usr/bin/env python3
"""
    Creates the training operation for a neural network
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
        Creates the operation to perform the Momentum optimization

        Args:
            loss: loss operation
            alpha: learning rate
            beta1: momentum weight

        Returns:
            the operation to perform the optimization
    """

    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
