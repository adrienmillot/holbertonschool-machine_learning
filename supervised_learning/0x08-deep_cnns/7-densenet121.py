#!/usr/bin/env python3
"""
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        Builds the DenseNet-121 architecture as described in Densely Connected
        Convolutional Networks

        Args:
            growth_rate: is an integer, the growth rate of the dense block
            compression: is a float between 0 and 1, the compression factor

        Returns:
            model: a Keras model
    """

    initializer = K.initializers.HeNormal()
    X = K.layers.Input(shape=(224, 224, 3))
    Y = K.layers.BatchNormalization()(X)
    Y = K.layers.Activation("relu")(Y)
    Y = K.layers.Conv2D(
        filters=2*growth_rate,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
    )(Y)
    Y = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(Y)

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=2*growth_rate,
        growth_rate=growth_rate,
        layers=6,
    )
    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=12,
    )
    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=24,
    )
    Y, nb_filters = transition_layer(
        X=Y,
        nb_filters=nb_filters,
        compression=compression,
    )

    Y, nb_filters = dense_block(
        X=Y,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=16,
    )

    Y = K.layers.AveragePooling2D(
        pool_size=7,
        padding="valid",
    )(Y)
    Y = K.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=initializer
    )(Y)

    return K.models.Model(inputs=X, outputs=Y)
