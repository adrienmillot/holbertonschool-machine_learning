#!/usr/bin/env python3
"""
    A simple neuron
"""

import numpy as np


class Neuron:
    """
        A simple neuron

        Attributes:
            __W: weights vector for the neuron.
            __b: bias for the neuron.
            __A: activated input for the neuron.
    """

    __W = 0
    __b = 0
    __A = 0

    def __init__(self, nx):
        """
            Constructor method.

            Args:
                nx: number of input features to the neuron.
        """

        if (type(nx) is not int):
            raise TypeError("nx must be an integer")

        if (nx < 1):
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)

    @property
    def W(self):
        """
            Getter method for weights vector
        """

        return self.__W

    @property
    def b(self):
        """
            Getter method for bias
        """

        return self.__b

    @property
    def A(self):
        """
            Getter method for activated input
        """

        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron.

            Args:
                X: input data.

            Returns:
                Activated output.
        """

        Z = np.dot(self.__W, X) + self.__b  # Initialize weight
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function

        return self.__A
