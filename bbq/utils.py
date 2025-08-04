"""Utility functions for the bbqudit package."""

import numpy as np
from bbq.field import Field


def cyclic_permutation(dim: int, shift: int) -> np.ndarray:
    """Construct cyclic shift permutation matrix of size dim x dim, shifted by shift."""

    if shift == 0:
        return np.identity(dim, dtype=int)
    else:
        return np.roll(np.identity(dim, dtype=int), shift, axis=1)


def err_to_det(h_eff: np.ndarray) -> dict:
    """
    Find the detectors and their stabiliser power in the neighbourhood of each error mechanism, i.e. {error : [(neighbouring detectors, power)]}.

    Parameters
    ----------
    h_eff : np.ndarray
        The effective parity check matrix.

    Returns
    -------
    err_neighbourhood : dict
        A dictionary mapping each error mechanism to a list of neighbouring detectors and their power (i.e. X, X^2, ...).
    """

    det, err = np.nonzero(h_eff)
    err_neighbourhood = {}
    for i in range(len(err)):
        if err[i] not in err_neighbourhood:
            err_neighbourhood[int(err[i])] = np.array(
                [(int(det[i]), int(h_eff[det[i], err[i]]))]
            )
        else:
            err_neighbourhood[int(err[i])] = np.vstack(
                [
                    err_neighbourhood[int(err[i])],
                    np.array([[int(det[i]), int(h_eff[det[i], err[i]])]]),
                ]
            )
    return err_neighbourhood


def det_to_err(h_eff: np.ndarray) -> dict:
    """
    Find the error mechanisms and their stabiliser power in the neighbourhood of each detector, i.e. {detector : [(neighbouring errors, power)]}.

    Parameters
    ----------
    h_eff : np.ndarray
        The effective parity check matrix.

    Returns
    -------
    det_neighbourhood : dict
        A dictionary mapping each detector to a list of neighbouring error mechanisms and their power (i.e. X, X^2, ...).
    """

    det, err = np.nonzero(h_eff)
    det_neighourhood = {}
    for i in range(len(det)):
        if det[i] not in det_neighourhood:
            det_neighourhood[int(det[i])] = np.array(
                [(int(err[i]), int(h_eff[det[i], err[i]]))]
            )
        else:
            det_neighourhood[int(det[i])] = np.vstack(
                [
                    det_neighourhood[int(det[i])],
                    np.array([[int(err[i]), int(h_eff[det[i], err[i]])]]),
                ]
            )
    return det_neighourhood


def rref(
    field: Field, A: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    """
    Perform Gaussian elimination on a linear system to find the reduced row echelon form (RREF) with pivots.

    Parameters
    ----------
    field : Field
        Field over which the matrix and vector are defined
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
        div = field.div(1, pivot)
        A_rref[row] = (A_rref[row] * div) % field.p
        v_rref[row] = (v_rref[row] * div) % field.p

        # Eliminate other elements in the pivot column
        for i in range(m):
            if i != row and A_rref[i, col] != 0:
                v_rref[i] -= A_rref[i, col] * v_rref[row]
                A_rref[i] -= A_rref[i, col] * A_rref[row]

        A_rref %= field.p
        v_rref %= field.p

        if len(pivot_rows) == m:
            break

    return (
        A_rref[sorted(pivot_rows)],
        v_rref[sorted(pivot_rows)],
        pivot_cols,
        pivot_rows,
        pivots,
    )
