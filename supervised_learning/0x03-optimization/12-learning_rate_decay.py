#!/usr/bin/env python3
"""
    Creates the operation to perform the learning rate decay
"""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Creates the operation to perform the learning rate decay

        Args:
            alpha: learning rate
            decay_rate: decay rate
            global_step: global step
            decay_step: decay step

        Returns:
            learning_rate: learning rate
    """

    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
