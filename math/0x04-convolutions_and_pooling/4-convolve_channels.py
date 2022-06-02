#!/usr/bin/env python3
"""
    Performs a convolution on images with channels
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on images with channels

        Args:
            images: np.ndarray with shape (m, h, w, c) containing multiple
                    images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
                c: number of channels of the images
            kernel: np.ndarray with shape (kh, kw) containing the kernel for
                    the convolution
                kh: kernel height
                kw: kernel width
            padding: string containing the type of padding. Can be 'same' or
                     'valid'
            stride: tuple with the strides for the convolution

        Returns:
            np.ndarray containing the convolved images
    """

    image_nb, image_height, image_width, _ = images.shape
    kernel_height, kernel_width, _ = kernel.shape
    stride_height, stride_width = stride
    image_size = np.arange(image_nb)

    # Calculate the padding height and width
    if padding == 'same':
        padding_height = int(
            (
                (
                    (image_height - 1) * stride_height +
                    kernel_height - image_height
                ) / 2
            ) + 1
        )
        padding_width = int(
            (
                (
                    (image_width - 1) * stride_width +
                    kernel_width - image_width
                ) / 2
            ) + 1
        )
    elif padding == 'valid':
        padding_height = 0
        padding_width = 0
    elif isinstance(padding, tuple):
        padding_height, padding_width = padding
    else:
        raise ValueError('Padding must be "same" or "valid"')

    # Calculate the output height and width
    convoluted_height = int(
        (image_height - kernel_height + 2 * padding_height) / stride[0] + 1
    )
    convoluted_width = int(
        (image_width - kernel_width + 2 * padding_width) / stride[1] + 1
    )

    # Add padding to the images
    padding = np.pad(
        images,
        pad_width=(
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
            (0, 0)
        ),
        mode='constant',
    )

    # Create a new matrice to hold the padded images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width)
    )

    for row in range(convoluted_height):
        for column in range(convoluted_width):
            input_image = padding[
                image_size,
                row * stride[0]: row * stride[0] + kernel_height,
                column * stride[1]: column * stride[1] + kernel_width,
            ]
            convoluted_images[image_size, row, column] = np.sum(
                input_image * kernel,
                axis=(1, 2, 3)
            )

    return convoluted_images
