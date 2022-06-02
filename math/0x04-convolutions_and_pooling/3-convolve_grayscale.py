#!/usr/bin/env python3
"""
    This script demonstrates how to use the convolve_grayscale function
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on grayscale images

        Args:
            images: np.ndarray with shape (m, h, w) containing multiple
                    grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
            kernel: np.ndarray with shape (kh, kw) containing the kernel
                    for the convolution
                kh: kernel height
                kw: kernel width
            padding: string containing the type of padding to be used
            stride: tuple with the strides for the convolution
            Returns: np.ndarray containing the convolved images
    """

    image_nb, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    stride_height, stride_width = stride
    img_size = np.arange(image_nb)

    if padding == 'same':
        padding_height = max(
            (image_height - 1) * stride[0] + kernel_height - image_height, 0
        )
        padding_width = max(
            (image_width - 1) * stride[1] + kernel_width - image_width, 0
        )
    elif padding == 'valid':
        padding_height = 0
        padding_width = 0
    else:
        padding_height, padding_width = padding

    convoluted_height = int(
        ((image_height + 2 * padding_height - kernel_height) / stride_height) + 1)
    convoluted_width = int(
        ((image_width + 2 * padding_width - kernel_width) / stride_width) + 1)

    padding = np.pad(images, ((0, 0), (padding_height, padding_height),
                     (padding_width, padding_width)), 'constant')

    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width))

    for row in range(convoluted_height):
        for colum in range(convoluted_width):
            s_row = row * stride_height
            s_column = colum * stride_width
            window = padding[img_size, s_row:kernel_height +
                             s_row, s_column:kernel_width+s_column]
            convoluted_images[img_size, row, colum] = np.sum(
                window * kernel, axis=(1, 2))
    return convoluted_images
