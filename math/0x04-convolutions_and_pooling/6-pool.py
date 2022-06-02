#!/usr/bin/env python3
"""
    Performs pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        Function that performs a pooling on images with channels

        Args:
            images: np.ndarray with shape (m, h, w, c) containing multiple
                    images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
                c: number of channels of the images
            kernel_shape: tuple with the kernel shape for the pooling
                kh: kernel height
                kw: kernel width
            stride: tuple with the strides for the pooling
                sh: stride height
                sw: stride width
            mode: string containing the type of pooling. Can be 'max' or 'avg'

        Returns:
            np.ndarray containing the pooled images
    """

    image_nb, image_height, image_width, image_channels = images.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = stride

    # Calculate the output height and width
    convoluted_height = int(
        (
            (image_height - kernel_height) / stride_height
        ) + 1
    )
    convoluted_width = int(
        (
            (image_width - kernel_width) / stride_width
        ) + 1
    )

    # Create a new matrice to hold the convoluted images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width, image_channels)
    )

    pooling = np.average if mode == 'avg' else np.max

    for column in range(convoluted_width):
        for row in range(convoluted_height):
            x = column * stride_width
            y = row * stride_height
            convoluted_images[:, row, column, :] = pooling(
                images[
                    :,
                    y: y + kernel_height,
                    x: x + kernel_width,
                    :
                ],
                axis=(1, 2),
            )

    return convoluted_images
