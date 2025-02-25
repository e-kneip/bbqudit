"""Implementation of the BivariateBicycle class for qudits."""

from polynomial import Polynomial
import numpy as np

class BivariateBicycle:
    """Implementation of the Bivariate Bicycle code on qudits.
    
    Parameters
    ----------
    a : Polynomial
        Polynomial a over the finite field.
    b : Polynomial
        Polynomial b over the finite field.
    l : int
        Dimension of left cyclic shift matrix.
    m : int
        Dimension of right cyclic shift matrix.
    q : int
        Defines CSS code construction H_x=(A|B) and H_y=(qB^T|(q-1)A^T).
    """

    def __init__(self, a : Polynomial, b : Polynomial, l : int, m : int, q : int):
        if not isinstance(a, Polynomial):
            raise TypeError("a must be a Polynomial")
        if not isinstance(b, Polynomial):
            raise TypeError("b must be a Polynomial")
        if not isinstance(l, int):
            raise TypeError("l must be an integer")
        if not isinstance(m, int):
            raise TypeError("m must be an integer")
        if not isinstance(q, int):
            raise TypeError("q must be an integer")
        if not 0 < q or not q < a.field:
            raise ValueError("q must be a positive integer less than the field of the polynomials")
        if a.field != b.field:
            raise ValueError("Polynomials a and b must be over the same field")
        self.a, self.b = a, b
        self.field = a.field
        self.l, self.m, self.q = l, m, q
        self.hx = np.hstack((a(l, m), b(l, m)))
        self.hy = np.hstack((q * b(l, m).transpose(), (self.field-q) * a(l, m).transpose())) % self.field

    def __str__(self):
        """String representation of BivariateBicycle."""
        return f"Bivariate Bicycle code for\na = {self.a}\nb = {self.b}"

    def __repr__(self):
        """Canonical string epresentation of BivariateBicycle."""
        return f"BivariateBicycle({self.a.__repr__()}, {self.b.__repr__()})"
