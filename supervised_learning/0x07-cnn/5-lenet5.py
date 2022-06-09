#!/usr/bin/env python3
"""
    Builds a modified version of the LeNet-5 architecture using keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
        Builds a modified version of the LeNet-5 architecture using keras

        Args:
            X: placeholder for the input data

        Returns:
            A LeNet-5 model
    """
    initializer = K.initializers.HeNormal()

    # Initialize the model
    model = K.Sequential()

    # Add the first convolutional layer
    model.add(
        K.layers.Conv2D(
            filters=6,
            kernel_size=5,
            activation="relu",
            padding='same',
            kernel_initializer=initializer,
        )
    )

    # Add the first pooling layer
    model.add(
        K.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
        )
    )

    # Add the second convolutional layer
    model.add(
        K.layers.Conv2D(
            filters=16,
            kernel_size=5,
            activation="relu",
            padding='valid',
            kernel_initializer=initializer,
        )
    )

    # Add the second pooling layer
    model.add(
        K.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
        )
    )

    # Flatten the output
    model.add(
        K.layers.Flatten()
    )

    # Add the first treatment layer
    model.add(
        K.layers.Dense(
            activation="relu",
            kernel_initializer=initializer,
            units=120,
        )
    )

    # Add the second treatment layer
    model.add(
        K.layers.Dense(
            activation="relu",
            kernel_initializer=initializer,
            units=84,
        )
    )

    # Add the output layer
    model.add(
        K.layers.Dense(
            activation='softmax',
            kernel_initializer=initializer,
            units=10,
        )
    )

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
