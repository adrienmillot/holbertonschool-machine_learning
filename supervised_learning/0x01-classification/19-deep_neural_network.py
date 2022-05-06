#!/usr/bin/env python3
"""
    A simple deep neural network
"""

import numpy as np


class DeepNeuralNetwork:
    """
        A simple deep neural network

        Attributes:
            L: number of layers.
            cache: dictionary to hold all intermediary values.
            weights: dictionary to hold all weights and biases.
    """

    __L = None
    __cache = {}
    __weights = {}

    def __init__(self, nx, layers):
        """
            Constructor method.

            Args:
                nx: number of input features to the neuron.
                layers: list representing the number of nodes in each layer.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        if any(list(map(lambda x: x <= 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)

        for layer_nb in range(self.__L):
            current_layer = layers[layer_nb]

            if layer_nb == 0:
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, nx
                ) * np.sqrt(2 / nx)

            else:
                previous_layer = layers[layer_nb - 1]
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, previous_layer
                ) * np.sqrt(2 / previous_layer)

            self.__weights["b" + str(layer_nb + 1)] = np.zeros(
                (current_layer, 1)
            )

    @property
    def L(self):
        """
            Getter method for the number of layers.
        """

        return self.__L

    @property
    def cache(self):
        """
            Getter method for the cache.
        """

        return self.__cache

    @property
    def weights(self):
        """
            Getter method for the weights.
        """

        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Args:
                X: numpy.ndarray (n, m) that contains the input data,
                    where n is the number of input features to the neuron
                    and m is the number of examples.
        """

        self.__cache["A0"] = X

        for layer_index in range(self.__L):
            b = self.__weights["b" + str(layer_index + 1)]
            a_previous = self.__cache["A" + str(layer_index)]
            w = self.__weights["W" + str(layer_index + 1)]
            # matmul function can't multiply matrices with different
            # dimensions.
            Z = np.dot(w, a_previous) + b
            # Sigmoid activation function.
            self.__cache["A" + str(layer_index + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y: numpy.ndarray (1, m) that contains the correct labels for
                    the input data.
                A: numpy.ndarray (1, m) containing the activated output of the
                    neuron for each example.
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
