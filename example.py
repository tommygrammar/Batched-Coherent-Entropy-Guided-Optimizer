"""
example.py

Demonstration script for the BCEGO_H optimizer on a simple 2D quadratic objective.

This script:
  1. Defines a two‐dimensional quadratic objective with a known maximum at (1,2).
  2. Initializes the BCEGO_H optimizer with a broad Gaussian prior.
  3. Runs the optimizer for a fixed number of iterations.
  4. Prints out the evolving belief (mean and covariance determinant),
     the best f-value each iteration, and flags when a refocus occurs.
"""

import numpy as np
from main import BCEGO_H  # Import the optimizer class

def fobj(X):
    """
    Quadratic objective function:
        f(x, y) = -[(x - 1)^2 + (y - 2)^2]

    This has a global maximum of 0 at (x, y) = (1, 2).

    Parameters
    ----------
    X : ndarray, shape (N, 2)
        Batch of N two-dimensional points.

    Returns
    -------
    fvals : ndarray, shape (N,)
        Objective values for each point.
    """
    return -((X[:, 0] - 1)**2 + (X[:, 1] - 2)**2)

# Problem dimensionality
dim = 2

# Initial Gaussian belief (mean at origin, broad covariance)
mu0 = np.zeros(dim)
Sigma0 = np.eye(dim) * 5.0

# Instantiate the BCEGO_H optimizer
opt = BCEGO_H(
    f                   = fobj,            # Objective function
    dim                 = dim,             # Dimensionality
    mu0                 = mu0,             # Initial mean
    Sigma0              = Sigma0,          # Initial covariance
    eta                 = 1.0,             # Fitness scaling factor
    batch_size          = 100,             # Samples per iteration
    tau_macro           = 0.6,             # Macro-cluster threshold
    tau_micro           = 0.85,            # Micro-cluster threshold
    refocus_thresh      = 1e-2,            # f-value threshold for refocus
    refocus_min_samples = 5,               # Minimum samples to trigger refocus
    random_seed         = 42               # Seed for reproducibility
)

# Run the optimizer for 50 iterations
history = opt.optimize(50)

# Print a summary of each iteration
for i, (mu, detS, best, refocused) in enumerate(history, 1):
    # Append a flag if this iteration triggered a refocus
    flag = " [REFOCUS]" if refocused else ""
    print(f"Iter {i:4d}: mean={mu}, detΣ={detS:.2e}, best f={best:.4f}{flag}")
