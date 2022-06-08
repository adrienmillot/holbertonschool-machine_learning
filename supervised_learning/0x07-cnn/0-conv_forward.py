#!/usr/bin/env python3
"""
    Performs forward propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        Performs forward propagation over a convolutional layer of
        a neural network

        Args:
            A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            W: numpy.ndarray of shape (f, f, c_prev, c_new)
                containing the kernels for the convolution
            b: numpy.ndarray of shape (1, 1, 1, c_new)
                containing the biases applied to the convolution
            activation: activation function used in the convolution
            padding: string that is either same or valid, indicating
                     the type of padding used
            stride: tuple of (sh, sw) containing the strides for
                    the convolution

        Returns:
            A_prev: numpy.ndarray containing the convolved output
            cache: cache used in the convolution
    """

    image_nb, image_height, image_width, _ = A_prev.shape
    filter_height, filter_width, _, filter_nb = W.shape
    stride_height, stride_width = stride

    # Calculate the padding height and width
    if padding == "same":
        padding_height = int(
            (
                (image_height - 1) * stride_height +
                filter_height - image_height
            ) / 2
        )
        padding_width = int(
            (
                (image_width - 1) * stride_width +
                filter_width - image_width
            ) / 2
        )
    elif padding == "valid":
        padding_height = 0
        padding_width = 0
    else:
        raise ValueError("padding must be valid or same")

    # Calculate the output height and width
    convoluted_height = int(
        (image_height + 2 * padding_height - filter_height) / stride_height
    ) + 1
    convoluted_width = int(
        (image_width + 2 * padding_width - filter_width) / stride_width
    ) + 1

    # Create a new matrice to hold the convoluted images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width, filter_nb))

    # Add padding to the images
    padding = np.pad(
        A_prev,
        (
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
            (0, 0)
        ),
        mode="constant"
    )

    # Perform the convolution
    kernels_cpy = W.copy()
    for row in range(convoluted_height):
        for column in range(convoluted_width):
            x = row * stride_height
            y = column * stride_width
            images_slide = padding[
                :,
                x:x + filter_height,
                y:y + filter_width,
                :
            ]

            for kernel_index in range(filter_nb):
                convoluted_images[:, row, column, kernel_index] = np.tensordot(
                    images_slide,
                    kernels_cpy[:, :, :, kernel_index],
                    axes=3
                ) + b[:, :, :, kernel_index]

    # Apply the activation function
    if activation is not None:
        convoluted_images = activation(convoluted_images)

    # Return the convoluted images
    return convoluted_images
