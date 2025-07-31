"""Implementation of a finite field."""

from sympy import isprime
import numpy as np


class Field:
    """A finite field of order p."""

    def __init__(self, p: int):
        """Initialise a finite field of order p."""

        if not isinstance(p, int):
            raise TypeError("p must be an integer")
        if not isprime(p):
            raise ValueError("p must be a prime number")

        self.p = p
        self._inverse = self._inverse()

    def _inverse(self) -> np.ndarray[int]:
        """Construct division table for a finite field, using brute force method."""
        # TODO: could use Euclid's extended algorithm (look at Finite field arithmetic Wikipedia page)
        table = np.zeros(self.p, dtype=int)
        table[1] = 1  # 1 is its own inverse
        for i in range(2, self.p):
            if not table[i]:
                for j in range(2, self.p):
                    if (i * j) % self.p == 1:
                        table[i] = j
                        table[j] = i
                        break
        return table

    def add(self, a: int, b: int) -> int:
        """Add two elements in the field."""
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        """Subtract two elements in the field."""
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        """Multiply two elements in the field."""
        return (a * b) % self.p

    def div(self, a: int, b: int) -> int:
        """Divide two elements in the field."""
        return (a * self._inverse[b]) % self.p
