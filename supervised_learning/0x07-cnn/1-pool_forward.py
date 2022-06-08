#!/usr/bin/env python3
"""
    Performs forward propagation over a pooling layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs forward propagation over a pooling layer of a neural network

        Args:
            A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            kernel_shape: tuple of (kh, kw) containing the size of the kernel
            stride: tuple of (sh, sw) containing the strides for the pooling
            mode: string containing the type of pooling, either max or average

        Returns:
            A: output of the pool layer
            cache: cache used in the pooling layer
    """

    image_nb, image_height, image_width, image_channels = A_prev.shape
    filter_height, filter_width = kernel_shape
    stride_height, stride_width = stride
    pooling = np.average if mode == 'avg' else np.max

    # Calculate the output height and width
    convoluted_height = int(
        (
            (image_height - filter_height) / stride_height
        ) + 1
    )
    convoluted_width = int(
        (
            (image_width - filter_width) / stride_width
        ) + 1
    )

    # Create a new matrice to hold the convoluted images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width, image_channels)
    )

    for column in range(convoluted_width):
        for row in range(convoluted_height):
            x = column * stride_width
            y = row * stride_height
            convoluted_images[:, row, column, :] = pooling(
                A_prev[
                    :,
                    y: y + filter_height,
                    x: x + filter_width,
                    :
                ],
                axis=(1, 2),
            )

    return convoluted_images
