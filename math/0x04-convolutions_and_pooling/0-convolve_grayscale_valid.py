#!/usr/bin/env python3
"""
    Performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        Function that performs a convolution on grayscale images with
        valid padding

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
            Returns: np.ndarray containing the convolved images
    """

    image_nb, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    image_size = np.arange(image_nb)

    # Calculate the convoluted height and width
    convoluted_height = image_height - kernel_height + 1
    convoluted_width = image_width - kernel_width + 1

    # Create a new matrice to hold the convoluted images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width)
    )

    # Perform the convolution
    for row in range(convoluted_height):
        for column in range(convoluted_width):
            input_image = images[
                image_size,
                row: row + kernel_height,
                column: column + kernel_width
            ]

            convoluted_images[image_size, row, column] = np.sum(
                input_image * kernel,
                axis=(1, 2)
            )

    return convoluted_images
