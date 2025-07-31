"""Implementation of a selection of decoders for qudits."""

import numpy as np
import galois
from numba import njit

from bbq.utils import err_to_det, det_to_err, rref, find_pivots
from bbq.field import Field


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


def _permute_field(field: Field) -> np.ndarray:
    """Construct permutations to shift errors according to stabiliser powers."""
    if field.p < 7:
        # For small fields, double for loop is faster than numpy
        permutation = np.zeros((field.p, field.p), dtype=int)
        for i in range(1, field.p):
            for j in range(1, field.p):
                permutation[i, j] = field.div(j, i)
        return permutation
    else:
        inv = field._inverse
        block = (np.arange(1, field.p)[np.newaxis, :] * inv[1:, np.newaxis]) % field.p
        return np.hstack(
            (
                np.zeros((field.p, 1), dtype=int),
                np.vstack((np.zeros((1, field.p - 1), dtype=int), block)),
            )
        )


# TODO: Don't worry about this yet
def _syn_inv_permute_field(syndrome: int, field: int) -> np.ndarray:
    """Construct permutations to shift errors according to syndrome and invert stabiliser powers."""
    permutation = np.zeros((field, field), dtype=int)
    for i in range(field):  # TODO: Never make this matrix
        for j in range(field):
            permutation[i, j] = (syndrome - j * i) % field
    return permutation


# TODO tip: D matrix  Dx = derivative
#           D @ x  -> loop i, j: (syndrome - j * i) % field * x[j]


@njit  # TODO: remove njit
def rearange_Q(Q_perm, errs, i, permutation):
    """Rearrange the error messages in Q according to the stabiliser powers."""
    for p in range(len(errs)):  # TODO: replace p with : and match axes
        Q_perm[errs[p, 0], i, :] = Q_perm[errs[p, 0], i, :][permutation[errs[p, 1], :]]
    return Q_perm


def _check_to_error_message(field, syndrome, P, Q, det_neighbourhood, permutation):
    """Pass messages from checks to errors."""
    for i, errs in det_neighbourhood.items():  # TODO: Deal with this later
        syn_inv_permutation = _syn_inv_permute_field(syndrome[i], field)

        # Permute elements in Q according to stabiliser powers
        Q_perm = Q.copy()
        for p in range(len(errs)):
            Q_perm[errs[p, 0], i, :] = Q_perm[errs[p, 0], i, :][
                permutation[errs[p, 1], :]
            ]
        # Q_perm = rearange_Q(Q_perm, errs, i, permutation)  # TODO: See this (fix rearange_Q then put into function as njit shouldn't help)

        # Fourier transform the relevant error messages
        convolution = np.fft.fft(Q_perm[errs[:, 0], i, :], axis=1)
        mask = np.ones(convolution.shape[0], dtype=bool)
        for j, error in enumerate(errs[:, 0]):
            # Remove the j-th error message from the convolution

            # create a mask that selects all rows except the j-th
            # This is equivalent to deleting the j-th row

            mask[j] = False

            # sub_convolution = np.delete(convolution, j, axis=0)
            sub_convolution = convolution[mask, :]

            # Compute the product of the transformed error messages
            sub_convolution = np.prod(sub_convolution, axis=0)

            # Inverse Fourier transform the product to find the subset convolution
            sub_convolution = np.fft.ifft(sub_convolution, axis=0).real

            # Pass message
            P[i, error, :] = sub_convolution[syn_inv_permutation[errs[j, 1], :]]
            # update the mask
            mask[j] = True

            # TODO: Find numpy function to compute masked products and lift out of loop
            #       lift ifft out of loop
            #       Store into P[i, :, :] == P[i, ...]


# TODO tip: np.einsum("j,i->ij", GF(np.arange(field)), 1 / GF(np.arange(1, field))) == np.arange(field)[np.newaxis, :] / GF(np.arange(1, field))[:, np.newaxis]
#           np.einsum(..., optimize=True)
def _error_to_check_message(prior, P, Q, err_neighbourhood):
    """Pass messages from errors to checks."""
    for (
        i,
        dets,
    ) in (
        err_neighbourhood.items()
    ):  # TODO: Vectorize this too (later) (consider using einsum)
        # Isolate the relevant check messages
        posterior = P[dets[:, 0], i, :]
        mask = np.ones(posterior.shape[0], dtype=bool)
        for j, detector in enumerate(dets[:, 0]):
            # Remove the j-th check message from the posterior
            # sub_posterior = np.delete(posterior, j, axis=0)
            mask[j] = False
            sub_posterior = posterior[mask, :]
            # Compute the product of probabilities
            sub_posterior = np.prod(sub_posterior, axis=0) * prior[i, :]

            #################################################
            # WARNING: sub_posterior is no longer normalised! <- check this?? I think I normalise in next line...
            #################################################

            # Pass normalised message
            Q[i, detector, :] = sub_posterior / np.sum(sub_posterior)

            # update the mask
            mask[j] = True

            # TODO: (This first!) Similar fix to _check_to_error_message


def _calculate_posterior(prior, n_errors, err_neighbourhood, P):
    """Calculate the posterior probabilities and make hard decision on error."""
    posteriors = np.zeros_like(prior)
    error = np.zeros(n_errors, dtype=int)

    for i, dets in err_neighbourhood.items():
        # TODO: Vectorize this:
        posterior = np.prod(P[dets[:, 0], i, :], axis=0) * prior[i, :]
        posterior /= np.sum(posterior) - posterior
        ####### do I have blowing up problems here??? yes, yes you do...
        posteriors[i, :] = posterior
        ############### does OSD want the likelihoods or the probabilities??? (I think likelihoods here)
        # TODO: Vectorize this ^

        # TODO: Do this as a separate operation (vectorized)
        max_lik = np.argmax(posterior)
        if posterior[max_lik] >= 1:  # Okay to keep this as a loop
            error[i] = max_lik

        ##############################################
        # WARNING: will pick lowest power error if there are 2 error types (eg X and X^2) that are likely (ONLY happens when eg likelihoods are (0, 1, 1) so doing > 1 instead of >= 1 would fix this *I think, possibly need > smaller number for higher field, eg (1/d)/(1-1/d)=1/d-1??*)
        ##############################################

    return error, posteriors


def belief_propagation(
    field: Field,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """Decode the syndrome using belief propagation.

    Parameters
    ----------
    field : Field
        The qudit dimension.
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
    error : nd.array
        The predicted error.
    success : bool
        Whether the decoding converged to a valid solution.
    """
    if not isinstance(field, Field):
        raise ValueError("field must be a Field instance")
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

    permutation = _permute_field(field)

    # Step 0: initialisation
    # Q[k, i] is the message passed from error k to check i
    Q = np.zeros((n_errors, n_detectors, field.p))
    for i in range(n_errors):
        #######################################################################
        # WARNING: If an error flags no detectors, sets messages to 0, => if syndrome is all 0, then will always say 0 errors (not a possible non-0 solution) *I think*
        #######################################################################

        # Send the same message of priors for each error to its neighbouring detectors
        if i in err_neighbourhood:
            Q[i, err_neighbourhood[i][:, 0], :] = prior[i]

    # P[i, k] is the message passed from check i to error k
    P = np.zeros((n_detectors, n_errors, field.p))

    for _ in range(max_iter):
        # Step 1: pass check to error messages
        _check_to_error_message(field.p, syndrome, P, Q, det_neighbourhood, permutation)

        # Step 2: pass error to check messages
        _error_to_check_message(prior, P, Q, err_neighbourhood)

        # Step 3: calculate posterior and make hard decision on errors
        error, posteriors = _calculate_posterior(prior, n_errors, err_neighbourhood, P)

        # Step 4: check convergence
        if np.all(h_eff @ error % field.p == syndrome):
            if debug:
                return error, True, True, posteriors
            else:
                return error, True

    if debug:
        return error, False, False, posteriors
    else:
        return error, False


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
    h_rref, syndrome_rref, pivot_cols, pivot_rows, pivots = rref(
        h_gf, syndrome_gf
    )  # may need pivots for qudits???
    rank = h_rref.shape[0]
    non_pivot_cols = [i for i in range(h_eff.shape[1]) if i not in pivot_cols]

    # Select the first rank(h_gf) linearly independent columns as basis set in P, others in B
    P = h_rref[:, pivot_cols]
    assert P.shape == (rank, rank)
    B = h_rref[:, non_pivot_cols]

    return P, B, rank, pivot_cols, non_pivot_cols, h_rref, syndrome_rref


def _rank_errors(  # TODO: Too many args
    g,
    field,
    n_errors,
    rank,
    h_rref,
    syndrome_rref,
    B,
    P,
    posterior,
    pivot_cols,  # TODO: Replace with a binary index and compute inside _rank_errors
    non_pivot_cols,  # TODO: Remove
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

    # TODO: error[pivot_cols] = fix
    #       p = posterior[pivot_cols, fix]
    # Then check if p > 0 ... (vectorized or keep the loop)

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
    # all of the for loop ranking above could be done with lambda functions???

    # Assert syndrome is satisfied
    assert (h_rref @ error == syndrome_rref).all()

    return np.array(error), score


def slow_osd(
    field: int,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    posterior: np.ndarray,
    certainties: np.ndarray = None,
    order: int = 0,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Decode the syndrome using an ordered statistics decoder (with Gaussian elimination).

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


def osd(
    field: Field,
    h_eff: np.ndarray,
    syndrome: np.ndarray,
    posterior: np.ndarray,
    certainties: np.ndarray = None,
    order: int = 0,
    debug: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Decode the syndrome using an ordered statistics decoder (with PLU decomposition).

    Parameters
    ----------
    field : Field
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
    if not isinstance(field, Field):
        raise ValueError("field must be a Field instance")
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
        # WARNING: Lose information here in the qudit case???
        certainties = np.sum(posterior[:, 1:], axis=1)

    n_detectors, n_errors = h_eff.shape
    GF = galois.GF(field.p)

    # Step 1: order the errors by likelihood
    permutation, inv_permutation = _find_permutation(certainties)
    h_eff = h_eff[:, permutation]

    # Step 2: decompose h_eff into rank(h_eff) linearly independent columns and rows (P) using plu decomposition
    # P, pivot_rows, pivot_cols = find_pivots(GF(h_eff))
    h_rref, syndrome_rref, pivot_cols, pivot_rows, pivots = rref(
        GF(h_eff), GF(syndrome)
    )
    P = h_eff[:, pivot_cols][pivot_rows, :]

    # Step 3: solve (wrt order) for the error mechanism with highest likelihood
    if order == 0:
        error = np.zeros(n_errors)
        syndrome_gf = GF(syndrome[pivot_rows])
        P_gf = GF(P)

        # Solve linear system
        short_error = np.linalg.solve(P_gf, syndrome_gf).__array__()
        error[pivot_cols] = short_error
        error = error[inv_permutation]
    else:
        raise NotImplementedError("OSD with order > 0 is not implemented yet.")

    # Invert permutation
    h_eff = h_eff[:, inv_permutation]

    assert ((h_eff @ error) % field.p == syndrome).all()

    if debug:
        return error, True, False, posterior
    else:
        return error, True


def d_osd(
    field: Field,
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
    field : Field
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
    if not isinstance(field, Field):
        raise ValueError("field must be a Field instance")
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
    field: Field,
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
    field : Field
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
    if not isinstance(field, Field):
        raise ValueError("field must be a Field instance")
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

    # Use sum of all likelihoods of X^k/Z^k errors on a  given qudit to rank h_eff columns
    # WARNING: Lose information here in the qudit case???
    certainties = np.sum(np.delete(posterior, 0, axis=1), axis=1)

    return osd(field, h_eff, syndrome, posterior, certainties, order, debug)


# TODO: Generate prior in advance in simulation, to be used in all shots
