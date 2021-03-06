#!/usr/bin/env python3
"""
    A simple deep neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions.

            Args:
                X: input data.
                Y: correct labels.

            Returns:
                The neuron's prediction and the cost of the network.
        """

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron.

            Args:
                Y: numpy.ndarray (1, m) that contains the correct labels for
                    the input data.
                cache: dictionary containing the output of each layer of the
                    neuron.
                alpha: learning rate.

            Returns:
                The updated weights of the neuron.
        """

        m = Y.shape[1]
        weights = self.__weights.copy()

        for layer_index in range(self.__L, 0, -1):
            A = cache["A" + str(layer_index)]

            if layer_index == self.__L:
                dz = A - Y
            else:
                dz = np.multiply(
                    np.dot(weights["W" + str(layer_index + 1)].T, dz),
                    A * (1 - A),
                )

            dw = 1 / m * np.dot(dz, cache["A" + str(layer_index - 1)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)

            self.__weights["W" + str(layer_index)] = weights[
                "W" + str(layer_index)] - alpha * dw
            self.__weights["b" + str(layer_index)] = weights[
                "b" + str(layer_index)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Trains the neuron.

            Args:
                X: input data.
                Y: correct labels.
                iterations: number of iterations to train.
                alpha: learning rate.

            Returns:
                The neuron's prediction and the cost of the network.
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")

            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x = []
        y = []

        for index in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose and index % step == 0:
                print("Cost after {} iterations: {}".format(
                    index, self.cost(Y, A)))

            if graph:
                x.append(index)
                y.append(self.cost(Y, A))

        if len(x) > 0 and len(y) > 0:
            plt.plot(x, y)
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format.
        """

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
            Loads the instance object from a file in pickle format.
        """

        try:
            if not filename.endswith(".pkl"):
                filename += ".pkl"

            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
