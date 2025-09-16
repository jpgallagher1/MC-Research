import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

def gauss_f(x):
    """Gaussian target distribution."""
    return np.exp(-x**2)
# Example target functions
def nongauss_f(x):
    """Non-gaussian target distribution."""
    return np.exp(-x**2 * (2 + np.sin(5*x) + np.sin(2*x)))

def gauss_ndimf(x, cov=None):
    """n-Dim Gaussian target distribution."""
    dim = len(x)
    if cov is None:
        cov = np.eye(dim)
    return np.exp(-x.dot(np.linalg.inv(cov).dot(x)))