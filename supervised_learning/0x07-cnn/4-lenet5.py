#!/usr/bin/env python3
"""
    Build a LeNet5 architecture
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        Build a LeNet5 architecture

        Args:
            x: placeholder for the input data
            y: placeholder for the input labels

        Returns:
            A LeNet5 model
    """

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Create a convolution layer
    convolutional_layer_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        activation=tf.nn.relu,
        padding='same',
        kernel_initializer=initializer,
    )(x)

    # Create a pooling layer
    pooling_layer_1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(convolutional_layer_1)

    # Create a convolution layer
    convolutional_layer_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        activation=tf.nn.relu,
        padding='valid',
        kernel_initializer=initializer,
    )(pooling_layer_1)

    # Create a pooling layer
    pooling_layer_2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(convolutional_layer_2)

    flatten = tf.layers.Flatten()(pooling_layer_2)

    # Create a treatement layer of 120 nodes
    treatment_layer_1 = tf.layers.Dense(
        activation=tf.nn.relu,
        units=120,
        kernel_initializer=initializer,
    )(flatten)

    # Create a treatement layer of 84 nodes
    treatment_layer_2 = tf.layers.Dense(
        activation=tf.nn.relu,
        units=84,
        kernel_initializer=initializer,
    )(treatment_layer_1)

    # Create an output layer of 10 nodes
    output_layer = tf.layers.Dense(
        units=10,
        kernel_initializer=initializer,
    )(treatment_layer_2)

    y_pred = tf.nn.softmax(output_layer)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=output_layer)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
