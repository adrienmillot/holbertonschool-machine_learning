#!/usr/bin/env python3
"""
    Computes L2 regularization cost with tensorflow
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
        Computes L2 regularization cost with tensorflow

        Args:
            cost: cost of the network without L2 regularization
            lambtha: regularization parameter
            weights: dictionary of the weights and biases of the neural network
            L: number of layers in the neural network
            m: number of data points

        Returns:
            cost: cost of the network with L2 regularization
    """

    return cost + tf.losses.get_regularization_losses()
