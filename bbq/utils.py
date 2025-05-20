"""Utility functions for the qudit-bivariate-bicycle package."""

import numpy as np


def cyclic_permutation(dim: int, shift: int) -> np.ndarray:
    """Construct cyclic shift permutation matrix of size dim x dim, shifted by shift."""

    if shift == 0:
        return np.identity(dim, dtype=int)
    else:
        return np.roll(np.identity(dim, dtype=int), shift, axis=1)


def err_to_det(h_eff: np.ndarray) -> dict:
    """
    Find the detectors in the neighbourhood of each error mechanism, i.e. {error : [neighbouring detectors]}.

    Parameters
    ----------
    h_eff : np.ndarray
        The effective parity check matrix.

    Returns
    -------
    err_neighbourhood : dict
        A dictionary mapping each error mechanism to a list of neighbouring detectors.
    """

    det, err = np.nonzero(h_eff)
    err_neighbourhood = {}
    for i in range(len(err)):
        if err[i] not in err_neighbourhood:
            err_neighbourhood[int(err[i])] = [int(det[i])]
        else:
            err_neighbourhood[int(err[i])].append(int(det[i]))
    return err_neighbourhood


def det_to_err(h_eff: np.ndarray) -> dict:
    """
    Find the error mechanisms in the neighbourhood of each detector, i.e. {detector : [neighbouring errors]}.

    Parameters
    ----------
    h_eff : np.ndarray
        The effective parity check matrix.

    Returns
    -------
    det_neighbourhood : dict
        A dictionary mapping each detector to a list of neighbouring error mechanisms.
    """

    det, err = np.nonzero(h_eff)
    det_neighourhood = {}
    for i in range(len(det)):
        if det[i] not in det_neighourhood:
            det_neighourhood[int(det[i])] = [int(err[i])]
        else:
            det_neighourhood[int(det[i])].append(int(err[i]))
    return det_neighourhood


def rref(A, v, x=None):
    """
    Perform Gaussian elimination to find the reduced row echelon form (RREF).
    Also identifies the pivot columns.
    Also reduces a vector to keep a linear system invariant.

    Parameters
    ----------
    A : Galois field array
        Galois field matrix to row reduce

    Returns
    -------
    A_rref : Galois field array
        Row-reduced form of A
    pivots : list
        Indices of pivot columns
    """
    # Get a copy to avoid modifying the original
    A_rref = A.copy()
    v_rref = v.copy()
    m, n = A_rref.shape
    assert v.shape == (m,)
    # assert (A_rref @ x == v_rref).all()

    # Track the pivot positions
    pivot_cols = []
    pivot_rows = []

    # Iterate through columns
    for c in range(n):
        # Find pivot in column c
        for r in range(m):
            if A_rref[r, c] != 0 and r not in pivot_rows:
                break
        else:
            continue

        # Record this column as a pivot column
        pivot_cols.append(c)
        pivot_rows.append(r)

        # Scale the pivot row to make the pivot element 1
        pivot = A_rref[r, c]
        A_rref[r] = A_rref[r] / pivot
        v_rref[r] = v_rref[r] / pivot

        # Eliminate other elements in the pivot column
        for i in range(m):
            if i != r and A_rref[i, c] != 0:
                v_rref[i] = v_rref[i] - A_rref[i, c] * v_rref[r]
                A_rref[i] = A_rref[i] - A_rref[i, c] * A_rref[r]

        # If we've exhausted all rows, we're done
        if len(pivot_rows) == m:
            break

    # if len(pivot_rows) < A.shape[0]:
    #     print("Matrix is not full rank.")

    return A_rref[sorted(pivot_rows)], v_rref[sorted(pivot_rows)], pivot_cols
