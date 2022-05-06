#!/usr/bin/env python3
"""
    A simple neuron
"""

import numpy as np
import matplotlib.pyplot as plt


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
        self.__A = 1 / (1 + np.exp(-Z))        # Sigmoid activation function

        return self.__A

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y: correct labels.
                A: activated output of the neuron for each example.

            Returns:
                The cost.
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

        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron.

            Args:
                X: input data.
                Y: correct labels.
                A: activated output of the neuron for each example.
                alpha: learning rate.

            Returns:
                The neuron's weights vector and bias.
        """

        m = X.shape[1]                    # Number of examples
        dz = A - Y
        dw = (1 / m) * np.matmul(X, dz.T)  # Gradient of the weights
        db = (1 / m) * np.sum(dz)         # Gradient of the bias

        self.__W -= (alpha * dw).T
        self.__b -= alpha * db

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
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

            if verbose and index % step == 0:
                print("Cost after {} iterations: {}".format(
                    index, self.cost(Y, self.__A)))

            if graph:
                x.append(index)
                y.append(self.cost(Y, self.__A))

        if len(x) > 0 and len(y) > 0:
            plt.plot(x, y)
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
