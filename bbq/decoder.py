"""Implementation of a selection of decoders for qudits."""

import numpy as np
import galois

from bbq.utils import err_to_det, det_to_err, rref


def dijkstra(h_eff: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
    """Order error mechanisms by distance to syndrome.

    Parameters
    ----------
    h_eff : nd.array
        The effective parity check matrix, where columns = error mechanism and rows = syndrome (flagged stabilisers).
    syndrome : nd.array
        The syndrome of the error.

    Returns
    -------
    error_distances : nd.array
        The distance of each error mechanism from a flagged detector.
    """
    if not isinstance(h_eff, np.ndarray):
        raise TypeError("h_eff must be a numpy array")
    if not isinstance(syndrome, np.ndarray):
        raise TypeError("syndrome must be a numpy array")

    m, n = h_eff.shape
    check_distances = np.ones(m) * (n + 1)
    error_distances = np.ones(n) * (n + 1)

    # Set the distance of flagged stabilisers to 0
    for c in syndrome.nonzero()[0]:
        check_distances[c] = 0

    # Set the distance each detector is from an error
    update_made = True
    while update_made:
        update_made = False
        for c in range(m):
            current_distance = check_distances[c]
            for e in np.nonzero(h_eff[c])[0]:
                if current_distance + 1 < error_distances[e]:
                    error_distances[e] = current_distance + 1
                    update_made = True

        for e in range(n):
            current_distance = error_distances[e]
            for c in np.nonzero(h_eff[:, e])[0]:
                if current_distance + 1 < check_distances[c]:
                    check_distances[c] = current_distance + 1
                    update_made = True

    return error_distances


def _check_to_error_message(field, syndrome, P, Q, det_neighbourhood):
    """Pass messages from checks to errors."""
    for i, errs in det_neighbourhood.items():
        # Fourier transform the relevant error messages
        convolution = np.fft.fft(Q[errs, i, :], axis=1)

        for j, error in enumerate(errs):
            # Remove the j-th error message from the convolution
            sub_convolution = np.delete(convolution, j, axis=0)

            # Compute the product of the transformed error messages
            sub_convolution = np.prod(sub_convolution, axis=0)

            # Inverse Fourier transform the product to find the subset convolution
            sub_convolution = np.fft.ifft(sub_convolution, axis=0)

            # Pass message
            for k in range(field):
                P[i, error, k] = sub_convolution[(syndrome[i] - k) % field]


def _error_to_check_message(prior, P, Q, err_neighbourhood):
    """Pass messages from errors to checks."""
    for i, dets in err_neighbourhood.items():
        # Isolate the relevant check messages
        posterior = P[dets, i, :]

        for j, detector in enumerate(dets):
            # Remove the j-th check message from the posterior
            sub_posterior = np.delete(posterior, j, axis=0)

            # Compute the product of probabilities
            sub_posterior = np.prod(sub_posterior, axis=0) * prior[i, :]

            #################################################
            # WARNING: sub_posterior is no longer normalised! <- check this?? I think I normalise in next line...
            #################################################

            # Pass normalised message
            Q[i, detector, :] = sub_posterior / np.sum(sub_posterior)


def _calculate_posterior(prior, n_errors, err_neighbourhood, P):
    """Calculate the posterior probabilities and make hard decision on error."""
    posteriors = np.zeros_like(prior.shape)
    error = np.zeros(n_errors, dtype=int)

    for i, dets in err_neighbourhood.items():
        posterior = np.prod(P[dets, i, :], axis=0) * prior[i, :]
        posterior /= np.sum(posterior) - posterior
        ####### do I have blowing up problems here??? yes, yes you do...
        posteriors[i, :] = posterior
        ############### does OSD want the likelihoods or the probabilities??? (I think likelihoods here)

        max_lik = np.argmax(posterior)
        if posterior[max_lik] >= 1:
            error[i] = max_lik

        ##############################################
        # WARNING: will pick lowest power error if there are 2 error types (eg X and X^2) that are likely (ONLY happens when eg likelihoods are (0, 1, 1) so doing > 1 instead of >= 1 would fix this *I think, possibly need > smaller number for higher field, eg (1/d)/(1-1/d)=1/d-1??*)
        ##############################################

    return error, posteriors


def belief_propagation(
    field: int,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """Decode the syndrome using belief propagation.

    Parameters
    ----------
    h_eff : nd.array
        The effective parity check matrix, where columns = error mechanism and rows = syndrome (flagged stabilisers).
    syndrome : nd.array
        The syndrome of the error.
    prior : nd.array
        The probability of each error mechanism.
    max_iter : int
        The maximum number of iterations, default is 1000.
    debug : bool
        Whether to return debug information (error, success, bp_success, posteriors), default is False.

    Returns
    -------
    error/posteriors : nd.array
        The predicted error if decoding successful, otherwise the posteriors.
    success : bool
        Whether the decoding converged to a valid solution.
    """
    if not isinstance(field, int) or field < 2:
        raise ValueError("field must be an integer greater than 1")
    if not isinstance(h_eff, np.ndarray):
        raise TypeError("h_eff must be a numpy array")
    if not isinstance(syndrome, np.ndarray):
        raise TypeError("syndrome must be a numpy array")
    if not isinstance(prior, np.ndarray):
        raise TypeError("prior must be a numpy array")
    if not (isinstance(max_iter, int) and max_iter > 0):
        raise ValueError("max_iter must be a positive integer")
    assert prior.shape[0] == h_eff.shape[1], (
        "prior must have the same number of entries as there are error mechanisms, i.e. columns of h_eff"
    )

    n_detectors, n_errors = h_eff.shape

    err_neighbourhood = err_to_det(h_eff)
    det_neighbourhood = det_to_err(h_eff)

    # Step 0: initialisation
    # Q[k, i] is the message passed from error k to check i
    Q = np.zeros((n_errors, n_detectors, field))
    for i in range(n_errors):
        #######################################################################
        # WARNING: If an error flags no detectors, sets messages to 0, => if syndrome is all 0, then will always say 0 errors (not a possible non-0 solution) *I think*
        #######################################################################

        # Send the same message of priors for each error to its neighbouring detectors
        if i in err_neighbourhood:
            Q[i, err_neighbourhood[i], :] = prior[i]

    # P[i, k] is the message passed from check i to error k
    P = np.zeros((n_detectors, n_errors, field))

    for _ in range(max_iter):
        # Step 1: pass check to error messages
        _check_to_error_message(field, syndrome, P, Q, det_neighbourhood)

        # Step 2: pass error to check messages
        _error_to_check_message(prior, P, Q, err_neighbourhood)

        # Step 3: calculate posterior and make hard decision on errors
        error, posteriors = _calculate_posterior(prior, n_errors, err_neighbourhood, P)

        # Step 4: check convergence
        if np.all(h_eff @ error % field == syndrome):
            if debug:
                return error, True, True, posteriors
            else:
                return error, True

    return posteriors, False


def _find_permutation(certainties):
    """Find the permutation of the error mechanisms based on their likelihood."""

    permutation = np.argsort(-certainties, axis=None)  # high certainty = low index
    inv_permutation = np.empty_like(permutation)
    inv_permutation[permutation] = np.arange(len(permutation))

    return permutation, inv_permutation


def _decompose(field, h_eff, syndrome):
    """Decompose h_eff into rank(h_eff) linearly independent columns (P) and the remainder (B) using rref."""

    # Convert to galois field
    GF = galois.GF(field)
    h_gf = GF(h_eff.copy())
    syndrome_gf = GF(syndrome.copy())

    # Find the reduced row echelon form (RREF) and identify pivot columns
    h_rref, syndrome_rref, pivot_cols, pivots = rref(
        h_gf, syndrome_gf
    )  # may need pivots for qudits???
    rank = h_rref.shape[0]
    non_pivot_cols = [i for i in range(h_eff.shape[1]) if i not in pivot_cols]

    # Select the first rank(h_gf) linearly independent columns as basis set in P, others in B
    P = h_rref[:, pivot_cols]
    assert P.shape == (rank, rank)
    B = h_rref[:, non_pivot_cols]

    return P, B, rank, pivot_cols, non_pivot_cols, h_rref, syndrome_rref


def _rank_errors(
    g,
    field,
    n_errors,
    rank,
    h_rref,
    syndrome_rref,
    B,
    P,
    posterior,
    pivot_cols,
    non_pivot_cols,
):
    """Calculate the error mechanism, satisfying the syndrome, with highest likelihood"""

    assert (
        g.shape == (n_errors - rank,)
    )  # could get rid of this assert if inputs are obvious (osd_0 yes, check for higher orders)
    GF = galois.GF(field)

    # Solve linear system
    remainder = syndrome_rref - B @ g
    fix = np.linalg.solve(P, remainder)
    assert (P @ fix + B @ g == syndrome_rref).all()

    # Rank the errors by likelihood
    score = 0
    error = GF.Zeros(n_errors)

    for i in range(rank):
        p = posterior[pivot_cols[i], fix[i]]
        error[pivot_cols[i]] = fix[i]
        if p > 0:
            score += np.log(p)
        else:
            score -= 1000

    for i in range(n_errors - rank):
        p = posterior[non_pivot_cols[i], g[i]]
        error[non_pivot_cols[i]] = g[i]
        if p > 0:
            score += np.log(p)
        else:
            score -= 1000

    # Assert syndrome is satisfied
    assert (h_rref @ error == syndrome_rref).all()

    return np.array(error), score


def osd(
    field: int,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    posterior: np.ndarray,
    certainties: np.ndarray = None,
    order: int = 0,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Decode the syndrome using an ordered statistics decoder.

    Parameters
    ----------
    field : int
        The qudit dimension.
    h_eff : nd.array
        The effective parity check matrix.
    syndrome : nd.array
        The syndrome of the error.
    posterior : nd.array
        The posterior probabilities of each error mechanism.
    certainties : nd.array
        The likelihoods of each error mechanism for ordering, default None constructs certainties from posterior.
    order : int
        The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
    debug : bool
        Whether to return debug information (error, success, pre_proccessing_success, posterior), default is False.

    Returns
    -------
    error : nd.array
        The predicted error mechanism.
    bool
        Whether the decoding was successful.
    """
    if not isinstance(field, int) or field < 2:
        raise ValueError("field must be an integer greater than 1")
    if not isinstance(h_eff, np.ndarray):
        raise TypeError("h_eff must be a numpy array")
    if not isinstance(syndrome, np.ndarray):
        raise TypeError("syndrome must be a numpy array")
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")
    if not isinstance(posterior, np.ndarray):
        raise TypeError("posterior must be a numpy array")
    if not (isinstance(certainties, np.ndarray) or certainties is None):
        raise TypeError("certainties must be a numpy array or None")

    if certainties is None:
        certainties = np.delete(posterior, 0, axis=1)
        # more complicated for qudits???

    n_detectors, n_errors = h_eff.shape
    GF = galois.GF(field)

    ####################################################################################
    # For qubits, do normal OSD (need to be careful with error powers for qudits *help*)
    ####################################################################################

    # Step 1: order the errors by likelihood
    permutation, inv_permutation = _find_permutation(certainties)
    h_eff = h_eff[:, permutation]
    posterior = posterior[
        permutation, :
    ]  # potentially want sth more complicated for qudits???

    # Step 2: decompose h_eff into rank(h_eff) linearly independent columns (P) and the remainder (B) using rref
    P, B, rank, pivot_cols, non_pivot_cols, h_rref, syndrome_rref = _decompose(
        field, h_eff, syndrome
    )

    # Step 3: solve (wrt order) for the error mechanism with highest likelihood
    if order == 0:
        error, score = _rank_errors(
            GF.Zeros(n_errors - rank),
            field,
            n_errors,
            rank,
            h_rref,
            syndrome_rref,
            B,
            P,
            posterior,
            pivot_cols,
            non_pivot_cols,
        )
        error = error[inv_permutation]
    else:
        raise NotImplementedError("OSD with order > 0 is not implemented yet.")

    # Invert permutation
    h_eff = h_eff[:, inv_permutation]
    posterior = posterior[inv_permutation, :]

    assert ((h_eff @ error) % field == syndrome).all()

    if debug:
        return error, True, False, posterior
    else:
        return error, True


def d_osd(
    field: int,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    prior: np.ndarray,
    order: int = 0,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Decode the syndrome using D+OSD (Dijkstra and Ordered Statistics Decoder).

    Parameters
    ----------
    field : int
        The qudit dimension.
    h_eff : nd.array
        The effective parity check matrix.
    syndrome : nd.array
        The syndrome of the error.
    prior : nd.array
        The prior probabilities of each error mechanism.
    order : int
        The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
    debug : bool
        Whether to return debug information (error, success, d_success, posteriors), default is False.

    Returns
    -------
    error : nd.array
        The predicted error mechanism.
    bool
        Whether the decoding was successful.
    """
    if not isinstance(field, int) or field < 2:
        raise ValueError("field must be an integer greater than 1")
    if not isinstance(h_eff, np.ndarray):
        raise TypeError("h_eff must be a numpy array")
    if not isinstance(syndrome, np.ndarray):
        raise TypeError("syndrome must be a numpy array")
    if not isinstance(prior, np.ndarray):
        raise TypeError("prior must be a numpy array")

    certainties = -dijkstra(
        h_eff, syndrome
    )  # negative for ordering: low distance = high likelihood
    return osd(field, h_eff, syndrome, prior, certainties, order, debug)


def bp_osd(
    field: int,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    order: int = 0,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Decode the syndrome using D+OSD (Dijkstra and Ordered Statistics Decoder).

    Parameters
    ----------
    field : int
        The qudit dimension.
    h_eff : nd.array
        The effective parity check matrix.
    syndrome : nd.array
        The syndrome of the error.
    prior : nd.array
        The prior probabilities of each error mechanism.
    max_iter : int
        The maximum number of iterations for belief propagation, default is 1000.
    order : int
        The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
    debug : bool
        Whether to return debug information (error, success, bp_success, posteriors), default is False.

    Returns
    -------
    error : nd.array
        The predicted error mechanism.
    bool
        Whether the decoding was successful.
    """
    if not isinstance(field, int) or field < 2:
        raise ValueError("field must be an integer greater than 1")
    if not isinstance(h_eff, np.ndarray):
        raise TypeError("h_eff must be a numpy array")
    if not isinstance(syndrome, np.ndarray):
        raise TypeError("syndrome must be a numpy array")
    if not isinstance(prior, np.ndarray):
        raise TypeError("prior must be a numpy array")
    if not (isinstance(max_iter, int) and max_iter > 0):
        raise ValueError("max_iter must be a positive integer")

    error, success, bp_success, posterior = belief_propagation(
        field, h_eff, syndrome, prior, max_iter, debug=True
    )
    if success:
        if debug:
            return error, success, bp_success, posterior
        else:
            return error, success

    # Qubit case: use only likelihood of X/Z error to rank h_eff columns, qudit case: tbd
    certainties = np.delete(posterior, 0, axis=1)

    return osd(field, h_eff, syndrome, posterior, certainties, order, debug)
