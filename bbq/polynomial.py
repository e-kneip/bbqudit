"""Implementation of the Polynomial class over finite fields."""

import numpy as np
from bbq.utils import cyclic_permutation
from bbq.field import Field
from numbers import Integral


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
                    if i == 0 and j == 0:
                        str_rep.append(f"{self.coefficients[i, j]}")
                    elif self.coefficients[i, j] > 1:
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

    def __eq__(self, other):
        """Check equality of two Polynomial instances."""
        if isinstance(other, Polynomial):
            return self.field == other.field and np.array_equal(
                self.coefficients, other.coefficients
            )
        return False

    def __ne__(self, other):
        """Check inequality of two Polynomial instances."""
        return not self.__eq__(other)

    def __add__(self, other):
        """Add two Polynomial instances or a Polynomial and an Integral."""
        if isinstance(other, Integral):
            coefs = self.coefficients.copy()
            coefs[0, 0] = (coefs[0, 0] + other) % self.field.p
            return Polynomial(self.field, coefs)
        if isinstance(other, Polynomial):
            if self.field != other.field:
                raise ValueError("Polynomials must be over the same field")
            max_x = max(self.coefficients.shape[0], other.coefficients.shape[0])
            max_y = max(self.coefficients.shape[1], other.coefficients.shape[1])

            self_coefs, other_coefs = (
                np.zeros((max_x, max_y), dtype=int),
                np.zeros((max_x, max_y), dtype=int),
            )
            self_coefs[: self.coefficients.shape[0], : self.coefficients.shape[1]] = (
                self.coefficients
            )
            other_coefs[
                : other.coefficients.shape[0], : other.coefficients.shape[1]
            ] = other.coefficients

            return Polynomial(self.field, (self_coefs + other_coefs) % self.field.p)
        return NotImplemented

    def __radd__(self, other):
        """Right addition for Polynomial and Integral."""
        if isinstance(other, Integral):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        """Subtract two Polynomial instances or a Polynomial and an Integral."""
        if isinstance(other, Integral):
            return self.__add__(-other % self.field.p)
        if isinstance(other, Polynomial):
            if self.field != other.field:
                raise ValueError("Polynomials must be over the same field")
            other_coefs = -other.coefficients.copy() % self.field.p
            return self.__add__(Polynomial(self.field, other_coefs))
        return NotImplemented

    def __rsub__(self, other):
        """Right subtraction for Polynomial and Integral."""
        if isinstance(other, Integral):
            self_coefs = -self.coefficients.copy() % self.field.p
            return Polynomial(self.field, self_coefs).__add__(other)
        return NotImplemented

    def __mul__(self, other):
        """Multiply a Polynomial instance by an Integral or Monomial."""
        if isinstance(other, Integral):
            coefs = self.coefficients.copy()
            coefs *= other
            return Polynomial(self.field, coefs % self.field.p)
        if isinstance(other, Monomial):
            if self.field != other.field:
                raise ValueError("Monomial must be over the same field as Polynomial")
            coefs = self.coefficients.copy()
            if other.monomial == "x":
                coefs = np.vstack(
                    (coefs, np.zeros((other.power, coefs.shape[1]), dtype=int))
                )
                coefs = np.roll(coefs, other.power, axis=0)
            elif other.monomial == "y":
                coefs = np.hstack(
                    (coefs, np.zeros((coefs.shape[0], other.power), dtype=int))
                )
                coefs = np.roll(coefs, other.power, axis=1)
            return Polynomial(self.field, coefs % self.field.p)
        if isinstance(other, Polynomial):
            if self.field != other.field:
                raise ValueError("Polynomials must be over the same field")
            product = 0
            locs = np.nonzero(self.coefficients)
            for x, y in zip(locs[0], locs[1]):
                prod = other * Monomial(self.field, "x", x)
                prod *= Monomial(self.field, "y", y)
                prod *= self.coefficients[x, y]
                product += prod
            return product
        return NotImplemented

    def __rmul__(self, other):
        """Right multiplication for Polynomial and Integral or Monomial."""
        if isinstance(other, Integral):
            return self.__mul__(other)
        if isinstance(other, Monomial):
            return other.__mul__(self)
        return NotImplemented

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


class Monomial(Polynomial):
    """Monomial class, a special case of Polynomial of the form: x^k or y^k."""

    def __init__(self, field, monomial, power=1):
        if not isinstance(field, Field):
            raise TypeError("field must be a Field instance")
        if not (monomial == "x" or monomial == "y"):
            raise ValueError("monomial must be 'x' or 'y'")
        if not isinstance(power, Integral):
            raise TypeError("power must be an Integral instance")
        self.monomial = monomial
        self.power = power

        if monomial == "x":
            coef = np.zeros((power + 1, 1), dtype=int)
            coef[power, 0] = 1
            super().__init__(field, coef)
        else:
            coef = np.zeros((1, power + 1), dtype=int)
            coef[0, power] = 1
            super().__init__(field, coef)

    def __pow__(self, power):
        """Raise the Monomial to a power."""
        if not isinstance(power, Integral):
            raise TypeError("power must be an Integral instance")
        if power < 0:
            raise ValueError("power must be non-negative")
        return Monomial(self.field, self.monomial, self.power * power)

    def __mul__(self, other):
        """Multiply Monomial by an Integral, Monomial or Polynomial."""
        if isinstance(other, Integral):
            return super().__mul__(other)
        elif isinstance(other, Monomial):
            if self.field != other.field:
                raise ValueError("Monomials must be over the same field")
            if self.monomial == other.monomial:
                new_power = self.power + other.power
                return Monomial(self.field, self.monomial, new_power)
            else:
                max_x = max(self.coefficients.shape[0], other.coefficients.shape[0])
                max_y = max(self.coefficients.shape[1], other.coefficients.shape[1])
                new_coefs = np.zeros((max_x, max_y), dtype=int)
                if self.monomial == "x":
                    new_coefs[self.power, other.power] = 1
                else:
                    new_coefs[other.power, self.power] = 1
                return Polynomial(self.field, new_coefs)
        elif isinstance(other, Polynomial):
            return other.__mul__(self)
        return NotImplemented

    def __rmul__(self, other):
        """Right multiplication for Monomial and Integral, Monomial or Polynomial."""
        if isinstance(other, Integral):
            return self.__mul__(other)
        elif isinstance(other, Monomial):
            return self.__mul__(other)
        elif isinstance(other, Polynomial):
            return other.__mul__(self)
        return NotImplemented
