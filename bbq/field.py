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

    def __repr__(self):
        """Canonical string representation of Field."""
        return f"Field({self.p})"

    def __eq__(self, other):
        """Check if two Field instances are equal."""
        if isinstance(other, Field):
            return self.p == other.p
        return False

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

    def rref(
        self, A: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[int]]:
        """
        Perform Gaussian elimination on a linear system to find the reduced row echelon form (RREF) with pivots.

        Parameters
        ----------
        A : np.ndarray
            Matrix to row reduce
        v : np.ndarray
            Vector to row reduce

        Returns
        -------
        A_rref : np.ndarray
            Row-reduced form of A
        v_rref : np.ndarray
            Row-reduced form of v
        pivot_cols : list[int]
            Indices of pivot columns
        pivot_rows : list[int]
            Indices of pivot rows
        pivots : list[int]
            Pivot values
        """

        A_rref = A.copy()
        v_rref = v.copy()
        m, n = A_rref.shape
        if not v.shape == (m,):
            raise ValueError(f"Expected v to have shape ({m},), got {v.shape}")

        # Track the pivot positions
        pivot_cols = []
        pivot_rows = []
        pivots = []

        # Iterate through columns
        for col in range(n):
            # Find pivot in column col
            for row in range(m):
                if A_rref[row, col] != 0 and row not in pivot_rows:
                    break
            else:
                continue

            pivot = int(A_rref[row, col])

            # Record the pivot
            pivot_cols.append(col)
            pivot_rows.append(row)
            pivots.append(pivot)

            # Scale the pivot row to make the pivot element 1
            div = self.div(1, pivot)
            A_rref[row] = (A_rref[row] * div) % self.p
            v_rref[row] = (v_rref[row] * div) % self.p

            # Eliminate other elements in the pivot column
            for i in range(m):
                if i != row and A_rref[i, col] != 0:
                    v_rref[i] -= A_rref[i, col] * v_rref[row]
                    A_rref[i] -= A_rref[i, col] * A_rref[row]

            A_rref %= self.p
            v_rref %= self.p

            if len(pivot_rows) == m:
                break

        return (
            A_rref[sorted(pivot_rows)],
            v_rref[sorted(pivot_rows)],
            pivot_cols,
            pivot_rows,
            pivots,
        )
