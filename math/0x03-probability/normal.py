#!/usr/bin/env python3
"""
    Probability normal representation module.
"""


class Normal:
    """
        Probability normal representation class.
    """

    mean = None
    stddev = None

    def __init__(self, data=None, mean=0., stddev=1.):
        """
            Constructor method.
        """

        # If data is given
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            self.stddev = pow(
                (sum(
                    [pow(x - self.mean, 2) for x in data]
                ) / len(data)), 0.5)

        # if data is not given
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """
            Z-score function.
        """

        if self.stddev == 0:
            return 0

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
            X-value function.
        """

        return self.mean + z * self.stddev

    def pdf(self, x):
        """
            Probability density function.
        """

        euler = 2.7182818285
        pi = 3.1415926536

        return pow(
            euler, -0.5 * pow(self.z_score(x), 2)
        ) / (self.stddev * pow(2 * pi, 0.5))

    def cdf(self, x):
        """
            Cumulative distribution function.
        """

        return 0.5 * (
            1 + self._erf((x - self.mean) / (self.stddev * pow(2, 0.5)))
        )

    def _erf(self, x):
        """
            Error function approximation
        """

        pi = 3.1415926536

        return 2 / pow(pi, 0.5) * (
            x - pow(x, 3)/3 + pow(x, 5)/10 - pow(x, 7)/42 + pow(x, 9)/216
        )
