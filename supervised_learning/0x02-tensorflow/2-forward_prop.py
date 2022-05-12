#!/usr/bin/env python3
"""
    Create the forward propagation graph
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph

        Args:
            x: placeholder for the input data
            layer_sizes: list containing the number of nodes in each layer
            activations: list containing the activation functions

        Returns:
            layer: tensorflow tensor object of the predicted labels
    """

    prev = x
    for index in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[index], activations[index])

    return prev
