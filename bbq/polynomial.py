"""Implementation of the Polynomial class over finite fields."""

import numpy as np
from bbq.utils import cyclic_permutation
from bbq.field import Field


class Polynomial:
    """Polynomial class over finite fields.

    Parameters
    ----------
    field : Field
        A Field instance defining the finite field.
    coefficients : np.ndarray
        Coefficients of the polynomial.
    """

    def __init__(self, field, coefficients):
        if not isinstance(field, Field):
            raise TypeError("field must be a Field instance")
        if not isinstance(coefficients, np.ndarray):
            raise TypeError("coefficients must be a ndarray")
        if coefficients.dtype != int:
            raise TypeError("coefficients must be an ndarray of integers")
        if coefficients.ndim != 2:
            raise ValueError("coefficients must be a 2D array")
        self.field = field
        self.coefficients = coefficients % field.p

    def __str__(self):
        """String representation of Polynomial."""
        str_rep = []
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                if self.coefficients[i, j] != 0:
                    if len(str_rep) > 0:
                        str_rep.append(" + ")
                    if self.coefficients[i, j] > 1:
                        str_rep.append(f"{self.coefficients[i, j]}")
                    if i == 1:
                        str_rep.append("x")
                    elif i > 1:
                        str_rep.append(f"x^{i}")
                    if j == 1:
                        str_rep.append("y")
                    elif j > 1:
                        str_rep.append(f"y^{j}")
        return "".join(str_rep if str_rep else "0")

    def __repr__(self):
        """Canonical string representation of Polynomial."""
        return f"Polynomial({self.field.__repr__()}, {self.coefficients.__repr__()})"

    def __call__(self, x_dim, y_dim):
        """Evaluate the Polynomial for cyclic shift permutation matrices of size x_dim, y_dim."""
        dim = self.coefficients.shape
        result = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                result.append(
                    self.coefficients[i, j]
                    * np.kron(
                        cyclic_permutation(x_dim, i), cyclic_permutation(y_dim, j)
                    )
                )
        return sum(result) % self.field.p

    def factor(self):
        """Find index of the lowest and highest degree, non-zero coefficient."""
        if (self.coefficients == 0).all():
            return np.array([0, 0])
        coef = self.coefficients
        coef_nonzero = coef.nonzero()
        min_ind = np.argmin(np.array(coef_nonzero).sum(axis=0))
        max_ind = np.argmax(np.array(coef_nonzero).sum(axis=0))
        min_coef_ind = np.array([coef_nonzero[0][min_ind], coef_nonzero[1][min_ind]])
        max_coef_ind = np.array([coef_nonzero[0][max_ind], coef_nonzero[1][max_ind]])
        return min_coef_ind, max_coef_ind
