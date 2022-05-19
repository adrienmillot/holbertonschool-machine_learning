#!/usr/bin/env python3
"""
    Performs the update with Adam optimization
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        Performs the update with Adam optimization

        Args:
            alpha: the learning rate
            beta1: the first Adam weight
            beta2: the second Adam weight
            epsilon: small number to avoid division by zero
            var: the variable to be updated
            grad: the gradient at the current step
            v: the previous first moment of var
            s: the previous second moment of var
            t: the time step

        Returns:
            the updated variable and the new v, s, and t
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    v_hat = v / (1 - beta1 ** t)
    s_hat = s / (1 - beta2 ** t)
    var = var - alpha * v_hat / (np.sqrt(s_hat) + epsilon)

    return var, v, s
