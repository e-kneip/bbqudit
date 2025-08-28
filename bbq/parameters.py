"""Calculate the parameters of a QuditCode."""

import numpy as np
import galois

from bbq.field import Field
from bbq.decoder import bp_osd


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
        print(log)
        dist = len(np.nonzero(log)[0])
        if dist < d:
            d = dist

    return d
