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

    L = None
    cache = {}
    weights = {}

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

        self.L = len(layers)

        for layer_nb in range(self.L):
            current_layer = layers[layer_nb]

            if layer_nb == 0:
                self.weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, nx
                ) * np.sqrt(2 / nx)

            else:
                previous_layer = layers[layer_nb - 1]
                self.weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, previous_layer
                ) * np.sqrt(2 / previous_layer)

            self.weights["b" + str(layer_nb + 1)] = np.zeros(
                (current_layer, 1)
            )
