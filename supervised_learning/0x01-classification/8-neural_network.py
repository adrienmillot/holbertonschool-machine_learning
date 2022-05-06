#!/usr/bin/env python3
"""
    A simple neural network
"""

import numpy as np


class NeuralNetwork:
    """
        A simple neural network

        Attributes:
            W1: weights vector for the neuron.
            b1: bias for the neuron.
            A1: activated input for the neuron.
            W2: weights vector for the neuron.
            b2: bias for the neuron.
            A2: activated input for the neuron.
    """

    W1 = None
    b1 = None
    A1 = None
    W2 = None
    b2 = None
    A2 = None

    def __init__(self, nx, nodes):
        """
            Constructor method.

            Args:
                nx: number of input features to the neuron.
                nodes: number of nodes in each layer.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
