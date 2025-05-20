"""Implementation of a selection of decoders for qudits."""

import numpy as np
import galois

from bbq.utils import err_to_det, det_to_err


def dijkstra(h, syndrome):
    """Order error mechanisms by distance to syndrome.

    Parameters
    ----------
    h : nd.array
        The effective parity check matrix, where columns = error mechanism and rows = syndrome (flagged stabilisers).
    syndrome : nd.array
        The syndrome of the error.
    prior : nd.array
        The probability of each error mechanism.

    Returns
    -------
    error_distances : nd.array
        The distance of each error mechanism from a flagged detector.
    """
    if not isinstance(h, np.ndarray):
        raise TypeError('h must be a numpy array')
    if not isinstance(syndrome, np.ndarray):
        raise TypeError('syndrome must be a numpy array')

    m, n = h.shape
    check_distances = np.ones(m) * (n+1)
    error_distances = np.ones(n) * (n+1)

    # Set the distance of flagged stabilisers to 0
    for c in syndrome.nonzero()[0]:
        check_distances[c] = 0

    # Set the distance each detector is from an error
    update_made = True
    while update_made:
        update_made = False
        for c in range(m):
            current_distance = check_distances[c]
            for e in np.nonzero(h[c])[0]:
                if current_distance + 1 < error_distances[e]:
                    error_distances[e] = current_distance + 1
                    update_made = True

        for e in range(n):
            current_distance = error_distances[e]
            for c in np.nonzero(h[:, e])[0]:
                if current_distance + 1 < check_distances[c]:
                    check_distances[c] = current_distance + 1
                    update_made = True

    return error_distances

def belief_propagation(field, h, syndrome, prior):
    if not isinstance(field, int) or field < 2:
        raise ValueError('field must be an integer greater than 1')
    if not isinstance(h, np.ndarray):
        raise TypeError('h must be a numpy array')
    if not isinstance(syndrome, np.ndarray):
        raise TypeError('syndrome must be a numpy array')
    if not isinstance(prior, np.ndarray):
        raise TypeError('prior must be a np.ndarray')

def osd():

    # Sort errors by how certain we are of their values (most to least)
    col_rank_perm = np.argsort(certainties)
    col_rank_inv_perm = np.empty_like(col_rank_perm)
    col_rank_inv_perm[col_rank_perm] = np.arange(len(col_rank_perm))

    # Create Galois field elements
    GF = galois.GF(field)
    
    # Convert H and syndrome to Galois field arrays
    H_gf = GF(h_eff.copy())
    syndrome_gf = GF(syndrome_history.copy())
    
    # Order the columns of h_gf and prior according to the ranking
    H_ordered_gf = H_gf[:, col_rank_perm]
    priors_perm = prior[col_rank_perm]

    # Find the reduced row echelon form (RREF) and identify pivot columns
    H_rref_gf, syndrome_rref_gf, pivot_cols = rref_with_pivots(H_ordered_gf, syndrome_gf)
    m_ind = H_rref_gf.shape[0]
    non_pivot_cols = [i for i in range(n) if i not in pivot_cols]

    # Select the first rank(h_gf) linearly independent columns as basis set in P, others in B
    P = H_rref_gf[:, pivot_cols]
    assert P.shape == (m_ind, m_ind)
    B = H_rref_gf[:, non_pivot_cols]

    def sln_from(g):
        assert g.shape == (n - m_ind,)
        remainder =  syndrome_rref_gf - B @ g
        fix = np.linalg.solve(P, remainder)
        assert (P @ fix + B @ g == syndrome_rref_gf).all()

        score = 0
        sln = GF.Zeros(n)
        # Find prob of basis set
        for i in range(m_ind):
            p = priors_perm[pivot_cols[i], fix[i]]
            sln[pivot_cols[i]] = fix[i]
            if p > 0:
                score += np.log(p)
            else:
                p -= 1000
        
        for i in range(n - m_ind):
            p = priors_perm[non_pivot_cols[i], g[i]]
            sln[non_pivot_cols[i]] = g[i]
            if p > 0:
                score += np.log(p)
            else:
                p -= 1000

        # Check sln makes correct syndrome
        assert (H_rref_gf @ sln == syndrome_rref_gf).all()
        assert (H_gf @ sln[col_rank_inv_perm] == syndrome_gf).all()
    
        assert ((h_eff @ sln[col_rank_inv_perm].__array__()) % field == syndrome_history).all()

        return np.array(sln[col_rank_inv_perm]), score

    # OSD_0 solution
    best_solution, best_score = sln_from(GF.Zeros(n - m_ind))
    assert ((h_eff @ best_solution) % field == syndrome_history).all()
    pivot_col_labels = {col_rank_perm[c]: int(error_distances[col_rank_perm[c]]) for c in pivot_cols}
    
    # print(f'for debugging: {best_solution}, {True}, {[col_rank_perm[i] for i in pivot_cols]}, {pivot_col_labels}')
    
    return best_solution

def d_osd():
    pass

def bp_osd():
    pass
