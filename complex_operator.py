import numpy as np
from scipy.special import gamma

def complex_binomial(a, k):
    return ((-1)**k) * gamma(a + 1) / (gamma(k + 1) * gamma(a - k + 1))

def compute_memory_velocity(history, alpha, beta, r):
    """
    Compute complex-order velocity memory term
    """
    a = complex(alpha, beta)
    memory = np.zeros_like(history[0])

    for k in range(min(r, len(history))):
        coeff = complex_binomial(a, k)
        memory += coeff.real * history[k]

    return memory