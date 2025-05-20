"""Utility functions for the qudit-bivariate-bicycle package."""

import numpy as np

def cyclic_permutation(dim : int, shift : int) -> np.ndarray:
    """Construct cyclic shift permutation matrix of size dim x dim, shifted by shift."""

    if shift == 0:
        return np.identity(dim, dtype=int)
    else:
        return np.roll(np.identity(dim, dtype=int), shift, axis=1)

def err_to_det(h_eff : np.ndarray) -> dict:
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

def det_to_err(h_eff : np.ndarray) -> dict:
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
