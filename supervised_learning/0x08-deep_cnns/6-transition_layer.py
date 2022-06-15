#!/usr/bin/env python3
"""
    Builds a transition layer as described in Densely Connected
    Convolutional Networks
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
        Builds a transition layer as described in Densely Connected
        Convolutional Networks

        Args:
            X: is the output from the previous layer
            nb_filters: is an integer, the number of filters in the convolution
            compression: is a float between 0 and 1, the compression factor

        Returns:
            X: the output of the transition layer
            nb_filters: the number of filters within the concatenated outputs
    """

    initializer = K.initializers.HeNormal()

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation("relu")(X)
    X = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer
    )(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2))(X)

    return X, int(nb_filters * compression)
