#!/usr/bin/env python3
"""
    Script that trains a convolutional neural network to classify
    the CIFAR 10 dataset
"""

import datetime
import numpy as np
import sys
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
        Preprocesses the data

        Args:
            X: numpy.ndarray of shape (m, 224, 224, 3)
                containing the training images
            Y: numpy.ndarray of shape (m, 224, 224, 1)
                containing the training labels

        Returns:
            X: numpy.ndarray of shape (m, 224, 224, 3)
                containing the training images
            Y: numpy.ndarray of shape (m, 224, 224, 1)
                containing the training labels
    """

    X_p = K.applications.inception_resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)

    return X_p, Y_p


if __name__ == '__main__':
    callbacks = []
    epochs = 5
    file_path = 'cifar10.h5'
    log_path = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    patience = 5

    # Load the data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the input data (test and train)
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Create base model
    base_model = K.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
    )

    # Freeze the model
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    input = K.layers.Lambda(lambda x: tf.image.resize(x, (128, 128)))(inputs)

    # Build the structure of layers
    X = base_model(input, training=False)
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Flatten()(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Dense(
        units=1000,
        activation="relu"
    )(X)
    X = K.layers.Dropout(0.5)(X)
    X = K.layers.BatchNormalization()(X)
    outputs = K.layers.Dense(
        units=500,
        activation="relu"
    )(X)
    X = K.layers.Dropout(0.2)(X)
    X = K.layers.BatchNormalization()(X)

    outputs = K.layers.Dense(
        units=10,
        activation="softmax"
    )(X)

    # Add early stopping callbacks
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        mode='min',
    )
    callbacks.append(early_stopping)

    # Add logs callbacks
    logs = K.callbacks.TensorBoard(
        log_dir=log_path,
        histogram_freq=1,
    )
    callbacks.append(logs)

    model = K.Model(inputs, outputs)

    # Configure the model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training the model
    model.fit(
        callbacks=callbacks,
        x=x_train,
        y=y_train,
        batch_size=128,
        epochs=epochs,
        shuffle=True,
        workers=15,
        validation_data=(x_test, y_test),
        use_multiprocessing=True,
    )

    model.trainable = True
    optimizer = K.optimizers.Adam(1e-5)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Training the model
    model.fit(
        callbacks=callbacks,
        x=x_train,
        y=y_train,
        batch_size=128,
        epochs=1,
        shuffle=True,
        workers=15,
        validation_data=(x_test, y_test),
        use_multiprocessing=True,
    )

    model.save(file_path)
