#!/usr/bin/env python3
"""
    A simple neural network
"""

import numpy as np


class NeuralNetwork:
    """
        A simple neural network

        Attributes:
            __W1: weights vector for the neuron.
            __b1: bias for the neuron.
            __A1: activated input for the neuron.
            __W2: weights vector for the neuron.
            __b2: bias for the neuron.
            __A2: activated input for the neuron.
    """

    __W1 = None
    __b1 = None
    __A1 = None
    __W2 = None
    __b2 = None
    __A2 = None

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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
            Getter method for weights vector
        """

        return self.__W1

    @property
    def b1(self):
        """
            Getter method for bias
        """

        return self.__b1

    @property
    def A1(self):
        """
            Getter method for activated input
        """

        return self.__A1

    @property
    def W2(self):
        """
            Getter method for weights vector
        """

        return self.__W2

    @property
    def b2(self):
        """
            Getter method for bias
        """

        return self.__b2

    @property
    def A2(self):
        """
            Getter method for activated input
        """

        return self.__A2

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron.

            Args:
                X: input data.
        """

        # matmul function can't multiply matrices with different dimensions.
        Z1 = np.dot(self.__W1, X) + self.__b1
        # Sigmoid activation function.
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        # Sigmoid activation function.
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y: correct labels vector.
                A: activated output of the neuron.
        """

        observations = Y.shape[1]
        precision = 1.0000001

        # Take the error when label=1
        class1_cost = -Y*np.log(A)

        # Take the error when label=0
        class2_cost = (1-Y)*np.log(precision-A)

        # Take the sum of both costs
        cost = class1_cost - class2_cost

        # Take the average cost
        cost = cost.sum() / observations

        return cost
