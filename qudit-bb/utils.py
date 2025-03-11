"""Utility functions for the qudit-bivariate-bicycle package."""

import numpy as np

def cyclic_permutation(dim : int, shift : int) -> np.ndarray:
    """Construct cyclic shift permutation matrix of size dim x dim, shifted by shift."""

    if shift == 0:
        return np.identity(dim, dtype=int)
    else:
        return np.roll(np.identity(dim, dtype=int), shift, axis=1)
