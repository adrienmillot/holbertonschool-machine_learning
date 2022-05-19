#!/usr/bin/env python3
"""
    Creates the operation to perform the RMSProp optimization
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the operation to perform the RMSProp optimization

        Args:
            loss: loss of the network
            alpha: learning rate
            beta2: decay rate
            epsilon: small number to avoid division by zero in RMSProp

        Returns:
            train_op: the RMSProp optimization operation
    """

    return tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    ).minimize(loss)
