#!/usr/bin/env python3
"""
    Probability exponential representation module.
"""


class Exponential:
    """
        Probability exponential representation class.
    """

    def __init__(self, data=None, lambtha=1.):
        """
            Constructor method
        """

        # If data is given
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(len(data) / sum(data))

        # if data is not given
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
            Probability density function.
        """

        if not isinstance(x, (int, float)):
            x = float(x)

        if x < 0:
            return 0

        euler = 2.7182818285

        return (self.lambtha * euler ** (-self.lambtha * x))

    def cdf(self, x):
        """
            Cumulative density function.
        """

        if not isinstance(x, (int, float)):
            x = float(x)

        if x < 0:
            return 0

        euler = 2.7182818285

        return (1 - euler ** (-self.lambtha * x))
