#!/usr/bin/env python3
"""
    Creates the forward propagation graph for the neural network using dropout
"""

import numpy as np


def softmax(Z):
    """
        Calculates the softmax of Z.

        Args:
            Z: numpy.ndarray (n, m) containing the input data

        Returns:
            A: numpy.ndarray (n, m) containing the softmax of Z
    """

    exp = np.exp(Z)
    return exp / exp.sum(axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Creates the forward propagation graph for the neural network
        using dropout

        Args:
            X: numpy.ndarray of shape (nx, m) that contains the input data
            weights: dictionary of the weights of the neural network
            L: The number of layers in the neural network
            keep_prob: probability that a node will be kept

        Returns:
            AL: numpy.ndarray of shape (1, m) containing the predictions
            cache: dictionary containing the outputs of each layer
    """

    cache = {}
    cache["A0"] = X

    for layer_index in range(L):
        a_previous = cache["A" + str(layer_index)]
        w = weights["W" + str(layer_index + 1)]
        b = weights["b" + str(layer_index + 1)]
        Z = np.dot(w, a_previous) + b

        if layer_index == L - 1:
            # Softmax Activation Function for Output Layer
            cache["A" + str(layer_index + 1)] = softmax(Z)
        else:
            drop = np.random.binomial(1, keep_prob, size=Z.shape)
            # Tanh Activation Function for Hidden layers
            cache["A" + str(layer_index + 1)] = np.tanh(Z) * drop / keep_prob
            cache["D" + str(layer_index + 1)] = drop

    return cache
