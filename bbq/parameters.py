"""Calculate the parameters of a QuditCode."""

import numpy as np
import galois

from bbq.field import Field
from bbq.decoder import bp_osd


def _ker_im(
    GF: galois._fields._meta.FieldArrayMeta, A: galois.FieldArray, B: galois.FieldArray
):
    """Compute elements in ker(A), excluding Im(B^T).

    Parameters
    ----------
    GF : galois._fields._meta.FieldArrayMeta
        The Galois field.
    A : galois.FieldArray
        The first Galois field array.
    B : galois.FieldArray
        The second Galois field array.

    Returns
    -------
    list[galois.FieldArray]
        The logical operators in ker(A) without Im(B^T).
    """

    check = B.copy()
    logicals = []

    ker_A = A.null_space()
    rank = np.linalg.matrix_rank(B)

    for vec in ker_A:
        check = GF(np.vstack((check, vec)))
        if np.linalg.matrix_rank(check) > rank:
            logicals.append(vec)
            rank += 1
        else:
            np.delete(check, -1, axis=0)

    return logicals


def logicals(field: Field, hx: np.ndarray, hz: np.ndarray):
    """Compute logical operators of a code.

    Parameters
    ----------
    field : Field
        The finite field the code is defined over.
    hx : np.ndarray
        The X parity check matrix.
    hz : np.ndarray
        The Z parity check matrix.

    Returns
    -------
    list[np.ndarray]
        The X logical operators.
    list[np.ndarray]
        The Z logical operators.
    """

    # Set up Galois field arrays
    GF = galois.GF(field.p)
    Hx_gal, Hz_gal = GF(hx), GF(hz)

    # X logicals must be in the kernel of Hz and not the image of Hx^T
    x_logicals = _ker_im(GF, Hz_gal, Hx_gal)

    # Z logicals must be in the kernel of Hx and not the image of Hz^T
    z_logicals = _ker_im(GF, Hx_gal, Hz_gal)

    # Check correct number of logicals found: k = n - m
    assert len(x_logicals) == len(z_logicals)
    m = np.linalg.matrix_rank(Hx_gal) + np.linalg.matrix_rank(Hz_gal)
    n = hx.shape[1]
    if not len(x_logicals) == n - m:
        raise ValueError("Incorrect number of logical operators found.")

    return (
        [x_log.__array__(dtype=int) for x_log in x_logicals],
        [z_log.__array__(dtype=int) for z_log in z_logicals],
    )


def bp_distance(
    field: Field,
    h: np.ndarray,
    logicals: np.ndarray,
    max_iter: int = 100,
    order: int = 0,
) -> int:
    """
    Approximate the distance of a code using belief propagation.

    Parameters
    ----------
    field : Field
        The finite field over which the code is defined.
    h : np.ndarray
        The parity check matrix of the code.
    logicals : list[np.ndarray]
        The logical operators of the code.
    max_iter : int
        The maximum number of iterations for belief propagation.
    order : int
        The order of OSD to use.

    Returns
    -------
    int
        The approximate distance of the code.
    """

    d = min(len(np.nonzero(logicals[i])[0]) for i in range(len(logicals)))

    syndrome = np.zeros(h.shape[0] + 1, dtype=int)
    syndrome[-1] = 1

    prior = np.zeros((h.shape[1], field.p))
    for i in range(field.p):
        prior[:, i] = 1 / field.p

    for logical in logicals:
        h_log = np.vstack((h, logical))
        log, _ = bp_osd(field, h_log, syndrome, prior, max_iter, order)
        dist = len(np.nonzero(log)[0])
        if dist < d:
            d = dist

    return d
