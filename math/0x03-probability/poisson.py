#!/usr/bin/env python3
"""
    Poisson module.
"""


class Poisson:
    """
        Poisson class.
    """

    _lambtha = None

    def __init__(self, data=None, lambtha=1.):
        """
            Constructor method.
        """

        # If data is given
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

        # if data is not given
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
            Probability mass function.
        """

        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0

        euler = 2.7182818285

        return (
            self.lambtha ** k * euler ** (-self.lambtha)
        ) / self._factorial(k)

    def _factorial(self, k):
        """
            Factorial function.
        """

        fact = 1

        for i in range(1, k + 1):
            fact *= i

        return fact

    def cdf(self, k):
        """
            Cumulative distribution function.
        """

        if type(k) is not int:
            k = int(k)

        if k < 0:
            return 0

        cdf = 0

        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf
