import numpy as np


def compute_eta_coefficients(alpha, beta, r):
    # Only 5 eta coefficients are defined in the paper (eta0 to eta4)
    if r > 5:
        raise ValueError(f"r={r} exceeds the 5 coefficients defined in paper.")
    a, b = alpha, beta
    eta = []

    #eta0
    eta.append(a)

    #eta1
    eta.append(-0.5 * (a**2 - a - b**2))

    #eta2
    eta.append(
        (1.0 / 6.0) * (a**3 - 3*a**2 + a*(2 - 3*b**2) + 3*b**2)
    )

    #eta3
    eta.append(
        (1.0 / 24.0) * (
            a**4 + 4*b**4 - 4*a**3 - 6*a**2*b**2
            + 11*(a**2 - b**2) + 18*a*b**2 - 6*a
        )
    )

    #eta4
    eta.append(
        (1.0 / 120.0) * (
            -a**5 + 10*a**4 + 10*a**3*b**2 - 35*a**3
            - 60*a**2*b**2 + 50*a**2 - 5*a*b**4
            + 105*a*b**2 - 24*a + 10*b**4 - 50*b**2
        )
    )

    #Return only r coefficients - paper sets r=4
    return eta[:r]


def compute_memory_velocity(history, alpha, beta, r):
    
    if len(history) == 0:
        raise ValueError("Velocity history is empty.")

    etas = compute_eta_coefficients(alpha, beta, r)
    memory = np.zeros_like(history[0], dtype=float)

    for k in range(min(r, len(history))):
        memory += etas[k] * history[k]

    return memory