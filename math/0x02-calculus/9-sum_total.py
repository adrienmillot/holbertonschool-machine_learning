#!/usr/bin/env python3
"""
    Square root of a number
"""


def summation_i_squared(n):
    """
        Returns the sum of the squares of the integers from 1 to n
    """

    if n <= 0:
        return None

    if n == 1:
        return 1

    return n * (n + 1) * (2 * n + 1) // 6
