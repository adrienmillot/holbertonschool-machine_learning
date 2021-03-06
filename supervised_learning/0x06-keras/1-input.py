#!/usr/bin/env python3
"""
    Builds a neural network with the Keras library.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Builds a neural network with the Keras library.

        Args:
            nx: is the number of input features to the neuron.
            layers: is a list representing the number of nodes in each layer.
            activations: is a list representing the activation functions for
                         each layer.
            lambtha: is the L2 regularization parameter.
            keep_prob: is the probability that a node will be kept.

        Returns:
            A Keras model instance.
    """

    # Create the input
    input = K.Input(shape=(nx,))

    for i in range(len(layers)):
        # Add the layer
        if i == 0:
            # first layer
            layer = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            )(input)
        else:
            # other layers
            layer = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            )(layer)

        # Add the dropout
        if i < len(layers) - 1:
            layer = K.layers.Dropout(1 - keep_prob)(layer)

    # Return the model
    return K.Model(inputs=input, outputs=layer)
