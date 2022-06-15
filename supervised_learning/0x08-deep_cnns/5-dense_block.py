#!/usr/bin/env python3
"""
    Builds a dense block as described in Densely Connected Convolutional
    Networks
"""
import tensorflow.keras as keras


def dense_block(X, nb_filters, growth_rate, layers):
    """
        Builds a dense block as described in Densely Connected Convolutional
        Networks

        Args:
            X: is the output from the previous layer
            nb_filters: is an integer, the number of filters in the convolution
            growth_rate: is an integer, the growth rate of the dense block
            layers: is an integer, the number of layers in the dense block

        Returns:
            X: the output of the dense block
            nb_filters: the number of filters within the concatenated outputs
    """

    initializer = keras.initializers.HeNormal()

    for i in range(layers):
        Y = keras.layers.BatchNormalization()(X)
        Y = keras.layers.Activation("relu")(Y)
        Y = keras.layers.Conv2D(
            filters=4*growth_rate,
            kernel_size=1,
            padding="same",
            kernel_initializer=initializer
        )(Y)
        Y = keras.layers.BatchNormalization()(Y)
        Y = keras.layers.Activation("relu")(Y)
        Y = keras.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding="same",
            kernel_initializer=initializer
        )(Y)

        X = keras.layers.concatenate([X, Y])
        nb_filters += growth_rate

    return X, nb_filters
