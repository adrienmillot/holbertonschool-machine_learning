#!/usr/bin/env python3
"""
    Calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
        Returns the integral of a polynomial
    """

    if (
        not isinstance(poly, list) or
        len(poly) == 0 or
        not isinstance(C, (int, float))
    ):
        return None

    if poly == [0]:
        return [C]

    integrate = [C]

    for index, value in enumerate(poly):
        if not isinstance(value, (int, float)):
            return None

        formula = value / (index + 1)
        integrate.append(int(formula) if formula.is_integer() else formula)

    return integrate
