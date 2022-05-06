#!/usr/bin/env python3
"""
    A simple neuron.
"""

import numpy as np


class Neuron:
    """
        A simple neuron.

        Attributes:
            W: weights vector for the neuron.
            b: bias for the neuron.
            A: activated input for the neuron.
    """

    W = 0
    b = 0
    A = 0

    def __init__(self, nx):
        """
            Constructor method.

            Args:
                nx: number of input features to the neuron.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
