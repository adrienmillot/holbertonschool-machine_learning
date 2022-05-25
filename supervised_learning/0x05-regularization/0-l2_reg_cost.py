#!/usr/bin/env python3
"""
    Computes L2 regularization cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Computes L2 regularization cost

        Args:
            cost: cost of the network without L2 regularization
            lambtha: regularization parameter
            weights: dictionary of the weights and biases of the neural network
            L: number of layers in the neural network
            m: number of data points

        Returns:
            cost: cost of the network with L2 regularization
    """
    sum_cost = 0

    for index in range(1, L + 1):
        sum_cost += np.sum(np.square(weights['W' + str(index)]))

    return cost + lambtha * (sum_cost / (2 * m))
