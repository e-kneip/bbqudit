"""Implementation of the Polynomial class over finite fields."""

import numpy as np
from utils import cyclic_permutation

class Polynomial:
    """Polynomial class over finite fields."""

    def __init__(self, field, coefficients):
        if type(field) is not int:
            raise TypeError("field must be an integer")
        if type(coefficients) is not np.ndarray:
            raise TypeError("coefficients must be a ndarray")
        if coefficients.dtype != int:
            raise TypeError("coefficients must be an ndarray of integers")
        if coefficients.ndim != 2:
            raise ValueError("coefficients must be a 2D array")
        self.field = field
        self.coefficients = coefficients % field

    def __str__(self):
        """String representation of Polynomial."""
        return " + ".join([f"{self.coefficients[i, j]}x^{i}y^{j}" for i in range(self.coefficients.shape[0]) for j in range(self.coefficients.shape[1])])

    def __repr__(self):
        """Canonical string representation of Polynomial."""
        return f"Polynomial({self.field}, {self.coefficients})"

    def __call__(self, x_dim, y_dim):
        """Evaluate the Polynomial for cyclic shift permutation matrices of size x_dim, y_dim."""
        dim = self.coefficients.shape
        result = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                result.append(self.coefficients[i, j] * np.kron(cyclic_permutation(x_dim, i), cyclic_permutation(y_dim, j)))
        return sum(result) % self.field
