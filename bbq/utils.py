"""Utility functions for the bbqudit package."""

import numpy as np
import numba


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
    det_neighbourhood = {}
    for i in range(len(det)):
        if det[i] not in det_neighbourhood:
            det_neighbourhood[int(det[i])] = np.array(
                [(int(err[i]), int(h_eff[det[i], err[i]]))]
            )
        else:
            det_neighbourhood[int(det[i])] = np.vstack(
                [
                    det_neighbourhood[int(det[i])],
                    np.array([[int(err[i]), int(h_eff[det[i], err[i]])]]),
                ]
            )
    return det_neighbourhood

@numba.njit
def norder(dim, order):
    """Generate all binary masks of errors with up to order non-zero entries."""
    masks = []
    mask = [0 for _ in range(dim)]
    for decimal in range(2**dim):
        index = 0
        for i in range(dim):
            val = decimal // (2**i) % 2
            mask[i] = val
            if val:
                index += 1
        if index > order:
            continue  # ideally would break out of two loops here :(
        masks.append(mask.copy())
    return np.array(masks)  # make sparse?
