#!/usr/bin/env python3
"""
    Calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
        Returns the derivative of a polynomial
    """

    if type(poly) is not list or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    return [
        poly[index]*index
        if isinstance(index, (int, float)) else None
        for index in range(1, len(poly))
    ]
