"""Implementation of a selection of decoders for qudits."""

from bbq.utils import err_to_det, det_to_err
from bbq.field import Field

from abc import ABC, abstractmethod
from numba import njit
import numpy as np


class Decoder(ABC):
    """Base class for decoders.
    
    Attributes
    ----------
    field : Field
        The qudit dimension.
    h : nd.array[int]
        The parity check matrix, where columns = error mechanism and rows = detectors.
    error_channel : nd.array[float]
        The probability of each error mechanism occuring.
    
    Methods
    -------
    decode(syndrome: nd.array[int]) -> nd.array[int]
        Decode the syndrome wrt the parity check matrix and error channel.
    """
    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float]):
        if not error_channel.shape == (h.shape[1], field.p):
            raise ValueError(
                "error_channel must have the same number of entries as there are error mechanisms, i.e. columns of h"
            )
        if not np.all(0 <= error_channel) & np.all(error_channel <= 1) & (np.isclose(np.sum(error_channel), 1) | np.all(np.sum(error_channel, axis=1) == 1)):
            raise ValueError("error_channel must be filled with probabilities, which sum to 1")
        field._validate(h)

        self.field = field
        self.h = h
        self.error_channel = error_channel

    @abstractmethod
    def decode(self, syndrome: np.ndarray[int]) -> tuple[np.ndarray[int], bool]:
        """Decode the syndrome wrt the parity check matrix and error channel.

        Parameters
        ----------
        syndrome : nd.array[int]
            The syndrome of the error.

        Returns
        -------
        error : nd.array[int]
            The predicted error mechanism.
        success : bool
            Whether the decoding was successful, i.e. if the syndrome is satisfied.
        """
        self.field._validate(syndrome)
        if not syndrome.shape == (self.h.shape[0],):
            raise ValueError(
                "syndrome must have the same number of entries as there are detectors, i.e. rows of h"
            )
        return np.zeros(self.h.shape[1], dtype=int), False


class Dijkstra(Decoder):
    """Decoder pre-processor (i.e. does not work stand alone) using Dijkstra's algorithm, ranking errors by their proximity to flagged detectors."""

    def __init__(self, field, h, error_channel):
        super().__init__(field, h, error_channel)

    def decode(self, syndrome: np.ndarray[int]) -> tuple[np.ndarray[int], bool]:
        """Rank likelihood of errors using Dijkstra's algorithm.

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.

        Returns
        -------
        error_distances : nd.array
            The distance of each error mechanism from a flagged detector.
        success : bool
            Success of decoder, always returns False as Dijkstra's algorithm is not a complete decoder.
        """
        super().decode(syndrome)

        m, n = self.h.shape
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
                for e in np.nonzero(self.h[c])[0]:
                    if current_distance + 1 < error_distances[e]:
                        error_distances[e] = current_distance + 1
                        update_made = True

            for e in range(n):
                current_distance = error_distances[e]
                for c in np.nonzero(self.h[:, e])[0]:
                    if current_distance + 1 < check_distances[c]:
                        check_distances[c] = current_distance + 1
                        update_made = True

        return error_distances, False


class BP(Decoder):
    """Decoder using belief propagation."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 1000):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations, default is 1000.
        """
        if not max_iter > 0:
            raise ValueError("max_iter must be a positive integer")

        super().__init__(field, h, error_channel)
        self.prior = error_channel
        self.max_iter = max_iter

        n_detectors, n_errors = self.h.shape

        self.err_neighbourhood = err_to_det(self.h)
        self.det_neighbourhood = det_to_err(self.h)
        self.permutation = self._permute_field()

        # Q[k, i] is the message passed from error k to check i
        self.Q = np.zeros((n_errors, n_detectors, self.field.p))
        for i in range(n_errors):
            #######################################################################
            # WARNING: If an error flags no detectors, sets messages to 0, => if syndrome is all 0, then will always say 0 errors (not a possible non-0 solution) *I think*
            #######################################################################

            # Send the same message of priors for each error to its neighbouring detectors
            if i in self.err_neighbourhood:
                self.Q[i, self.err_neighbourhood[i][:, 0], :] = self.prior[i]

        # P[i, k] is the message passed from check i to error k
        self.P = np.zeros((n_detectors, n_errors, self.field.p))

    def _permute_field(self) -> np.ndarray:
        """Construct permutations to shift errors according to stabiliser powers."""
        if self.field.p < 7:
            # For small fields, double for loop is faster than numpy
            permutation = np.zeros((self.field.p, self.field.p), dtype=int)
            for i in range(1, self.field.p):
                for j in range(1, self.field.p):
                    permutation[i, j] = self.field.div(j, i)
            return permutation
        else:
            inv = self.field._inverse
            block = (np.arange(1, self.field.p)[np.newaxis, :] * inv[1:, np.newaxis]) % self.field.p
            return np.hstack(
                (
                    np.zeros((self.field.p, 1), dtype=int),
                    np.vstack((np.zeros((1, self.field.p - 1), dtype=int), block)),
                )
            )


    # TODO: Don't worry about this yet
    def _syn_inv_permute_field(self, syndrome: int) -> np.ndarray:
        """Construct permutations to shift errors according to syndrome and invert stabiliser powers."""
        permutation = np.zeros((self.field.p, self.field.p), dtype=int)
        for i in range(self.field.p):  # TODO: Never make this matrix
            for j in range(self.field.p):
                permutation[i, j] = (syndrome - j * i) % self.field.p
        return permutation


    # TODO tip: D matrix  Dx = derivative
    #           D @ x  -> loop i, j: (syndrome - j * i) % field * x[j]


    @njit
    def rearange_Q(self, Q_perm, errs, i, permutation):
        """Rearrange the error messages in Q according to the stabiliser powers."""
        for p in range(len(errs)):
            Q_perm[errs[p, 0], i, :] = Q_perm[errs[p, 0], i, :][permutation[errs[p, 1], :]]
        return Q_perm


    def _check_to_error_message(self, syndrome, P, Q):
        """Pass messages from checks to errors."""
        for i, errs in self.det_neighbourhood.items():  # TODO: Deal with this later
            syn_inv_permutation = self._syn_inv_permute_field(syndrome[i])

            # Permute elements in Q according to stabiliser powers
            Q_perm = Q.copy()
            Q_perm[errs[:, 0], i, :] = np.take_along_axis(
                Q_perm[errs[:, 0], i, :], self.permutation[errs[:, 1], :], axis=1
            )
            # Q_perm = self.rearange_Q(Q_perm, errs, i, permutation) (NOTE: older code, is about the same speed as the above line but maybe slower for larger simulations?)

            # Fourier transform the relevant error messages
            convolution = np.fft.fft(Q_perm[errs[:, 0], i, :], axis=1)

            # Compute the product of the probabilities for the error messages, excluding one row of messages to avoid feedback
            sub_convolutions = np.prod(convolution, axis=0)
            sub_convolutions = sub_convolutions / convolution

            # Inverse Fourier transform the product to find the subset convolution
            sub_convolution = np.fft.ifft(sub_convolutions, axis=1).real

            # Pass message
            P[i, errs[:, 0], :] = np.take_along_axis(
                sub_convolution, syn_inv_permutation[errs[:, 1], :], axis=1
            )


    # TODO tip: np.einsum("j,i->ij", GF(np.arange(field)), 1 / GF(np.arange(1, field))) == np.arange(field)[np.newaxis, :] / GF(np.arange(1, field))[:, np.newaxis]
    #           np.einsum(..., optimize=True)
    def _error_to_check_message(self, P, Q):
        """Pass messages from errors to checks."""
        for i, dets in self.err_neighbourhood.items():
            # TODO: Vectorize this too (later) (consider using einsum)

            # Isolate the relevant check messages
            posterior = P[dets[:, 0], i, :]

            sub_posteriors = np.prod(posterior, axis=0) * self.prior[i, :]
            sub_posteriors = sub_posteriors / posterior

            # Pass normalised messages
            Q[i, dets[:, 0], :] = (
                sub_posteriors / np.sum(sub_posteriors, axis=1)[:, np.newaxis]
            )


    def _calculate_posterior(self, P):
        """Calculate the posterior probabilities and make hard decision on error."""
        posteriors = np.zeros_like(self.prior)

        errs = list(self.err_neighbourhood.keys())
        posteriors[errs, :] = (
            np.array([np.prod(P[self.err_neighbourhood[i][:, 0], i, :], axis=0) for i in errs])
            * self.prior[errs, :]
        )

        for i, dets in self.err_neighbourhood.items():
            # TODO: Vectorize this:
            posterior = np.prod(P[dets[:, 0], i, :], axis=0) * self.prior[i, :]
            posteriors[i, :] = posterior

        posteriors /= (
            np.sum(posteriors, axis=1)[:, np.newaxis] - posteriors
        )  ####### do I have blowing up problems here??? yes, yes you do...

        max_lik = np.argmax(posteriors, axis=1)
        # if 50:50 chance between 2 errors, max_lik will pick the 1st in row (lower power)
        error = np.array(
            [
                max_lik[i] if posteriors[i, max_lik[i]] >= 1 else 0
                for i in range(posteriors.shape[0])
            ]
        )

        return error, posteriors


    def decode(self, syndrome: np.ndarray, debug: bool = False) -> tuple[np.ndarray[int], bool]:
        """Decode the syndrome using belief propagation.

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False. Use if post-processing results.

        Returns
        -------
        error : nd.array
            The predicted error.
        success : bool
            Whether the decoding converged to a valid solution.
        """
        if not isinstance(syndrome, np.ndarray):
            raise TypeError("syndrome must be a numpy array")

        P, Q = self.P.copy(), self.Q.copy()

        for _ in range(self.max_iter):
            # Pass messages
            self._check_to_error_message(syndrome, P, Q)
            self._error_to_check_message(P, Q)

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(P)

            # Check convergence
            if np.all(self.h @ error % self.field.p == syndrome):
                if debug:
                    return error, True, True, posteriors
                else:
                    return error, True

        if debug:
            return error, False, False, posteriors
        else:
            return error, False


class OSD(Decoder):
    """Decoder using ordered statistics decoding."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], posterior: np.ndarray[float], certainties: np.ndarray[float] | None = None, order: int = 0):
        """
        Initialise an ordered statistics decoder.
        
        Parameters
        ----------
        posterior : nd.array
            The posterior probabilities of each error mechanism.
        certainties : nd.array
            The likelihoods of each error mechanism for ordering, default None constructs certainties from posterior.
        order : int
            The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
        """
        if not order >= 0:
            raise ValueError("order must be a non-negative integer")
        
        super().__init__(field, h, error_channel)
        self.posterior = posterior
        self.certainties = certainties
        self.order = order

    def _find_permutation(self, certainties):
        """Find the permutation of the error mechanisms based on their likelihood."""

        permutation = np.argsort(-certainties, axis=None)  # high certainty = low index
        inv_permutation = np.empty_like(permutation)
        inv_permutation[permutation] = np.arange(len(permutation))

        return permutation, inv_permutation


    # def _decompose(field, h_eff, syndrome):
    #     """Decompose h_eff into rank(h_eff) linearly independent columns (P) and the remainder (B) using rref."""

    #     # Convert to galois field
    #     GF = galois.GF(field)
    #     h_gf = GF(h_eff.copy())
    #     syndrome_gf = GF(syndrome.copy())

    #     # Find the reduced row echelon form (RREF) and identify pivot columns
    #     h_rref, syndrome_rref, pivot_cols, pivot_rows, pivots = rref(
    #         h_gf, syndrome_gf
    #     )  # may need pivots for qudits???
    #     rank = h_rref.shape[0]
    #     non_pivot_cols = [i for i in range(h_eff.shape[1]) if i not in pivot_cols]

    #     # Select the first rank(h_gf) linearly independent columns as basis set in P, others in B
    #     P = h_rref[:, pivot_cols][pivot_rows, :]
    #     assert P.shape == (rank, rank)
    #     B = h_rref[:, non_pivot_cols]

    #     return P, B, rank, pivot_cols, non_pivot_cols, h_rref, syndrome_rref


    # def _rank_errors(  # TODO: Too many args
    #     g,
    #     field,
    #     n_errors,
    #     rank,
    #     h_rref,
    #     syndrome_rref,
    #     B,
    #     P,
    #     posterior,
    #     pivot_cols,  # TODO: Replace with a binary index and compute inside _rank_errors
    #     non_pivot_cols,  # TODO: Remove
    # ):
    #     """Calculate the error mechanism, satisfying the syndrome, with highest likelihood"""

    #     assert (
    #         g.shape == (n_errors - rank,)
    #     )  # could get rid of this assert if inputs are obvious (osd_0 yes, check for higher orders)
    #     GF = galois.GF(field)

    #     # Solve linear system
    #     remainder = syndrome_rref - B @ g
    #     fix = np.linalg.solve(P, remainder)
    #     assert (P @ fix + B @ g == syndrome_rref).all()

    #     # Rank the errors by likelihood
    #     score = 0
    #     error = GF.Zeros(n_errors)

    #     # TODO: error[pivot_cols] = fix
    #     #       p = posterior[pivot_cols, fix]
    #     # Then check if p > 0 ... (vectorized or keep the loop)

    #     for i in range(rank):
    #         p = posterior[pivot_cols[i], fix[i]]
    #         error[pivot_cols[i]] = fix[i]
    #         if p > 0:
    #             score += np.log(p)
    #         else:
    #             score -= 1000

    #     for i in range(n_errors - rank):
    #         p = posterior[non_pivot_cols[i], g[i]]
    #         error[non_pivot_cols[i]] = g[i]
    #         if p > 0:
    #             score += np.log(p)
    #         else:
    #             score -= 1000
    #     # all of the for loop ranking above could be done with lambda functions???

    #     # Assert syndrome is satisfied
    #     assert (h_rref @ error == syndrome_rref).all()

    #     return np.array(error), score


    # def slow_osd(
    #     field: int,
    #     h_eff: np.ndarray,
    #     syndrome: np.ndarray,
    #     posterior: np.ndarray,
    #     certainties: np.ndarray = None,
    #     order: int = 0,
    #     debug: bool = False,
    # ) -> tuple[np.ndarray, bool]:
    #     """
    #     Decode the syndrome using an ordered statistics decoder (with Gaussian elimination).

    #     Parameters
    #     ----------
    #     field : int
    #         The qudit dimension.
    #     h_eff : nd.array
    #         The effective parity check matrix.
    #     syndrome : nd.array
    #         The syndrome of the error.
    #     posterior : nd.array
    #         The posterior probabilities of each error mechanism.
    #     certainties : nd.array
    #         The likelihoods of each error mechanism for ordering, default None constructs certainties from posterior.
    #     order : int
    #         The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
    #     debug : bool
    #         Whether to return debug information (error, success, pre_proccessing_success, posterior), default is False.

    #     Returns
    #     -------
    #     error : nd.array
    #         The predicted error mechanism.
    #     bool
    #         Whether the decoding was successful.
    #     """
    #     if not isinstance(field, int) or field < 2:
    #         raise ValueError("field must be an integer greater than 1")
    #     if not isinstance(h_eff, np.ndarray):
    #         raise TypeError("h_eff must be a numpy array")
    #     if not isinstance(syndrome, np.ndarray):
    #         raise TypeError("syndrome must be a numpy array")
    #     if not isinstance(order, int) or order < 0:
    #         raise ValueError("order must be a non-negative integer")
    #     if not isinstance(posterior, np.ndarray):
    #         raise TypeError("posterior must be a numpy array")
    #     if not (isinstance(certainties, np.ndarray) or certainties is None):
    #         raise TypeError("certainties must be a numpy array or None")

    #     if certainties is None:
    #         certainties = np.delete(posterior, 0, axis=1)
    #         # more complicated for qudits???

    #     n_detectors, n_errors = h_eff.shape
    #     GF = galois.GF(field)

    #     ####################################################################################
    #     # For qubits, do normal OSD (need to be careful with error powers for qudits *help*)
    #     ####################################################################################

    #     # Step 1: order the errors by likelihood
    #     permutation, inv_permutation = _find_permutation(certainties)
    #     h_eff = h_eff[:, permutation]
    #     posterior = posterior[
    #         permutation, :
    #     ]  # potentially want sth more complicated for qudits???

    #     # Step 2: decompose h_eff into rank(h_eff) linearly independent columns (P) and the remainder (B) using rref
    #     P, B, rank, pivot_cols, non_pivot_cols, h_rref, syndrome_rref = _decompose(
    #         field, h_eff, syndrome
    #     )

    #     # Step 3: solve (wrt order) for the error mechanism with highest likelihood
    #     if order == 0:
    #         error, score = _rank_errors(
    #             GF.Zeros(n_errors - rank),
    #             field,
    #             n_errors,
    #             rank,
    #             h_rref,
    #             syndrome_rref,
    #             B,
    #             P,
    #             posterior,
    #             pivot_cols,
    #             non_pivot_cols,
    #         )
    #         error = error[inv_permutation]
    #     else:
    #         raise NotImplementedError("OSD with order > 0 is not implemented yet.")

    #     # Invert permutation
    #     h_eff = h_eff[:, inv_permutation]
    #     posterior = posterior[inv_permutation, :]

    #     assert ((h_eff @ error) % field == syndrome).all()

    #     if debug:
    #         return error, True, False, posterior
    #     else:
    #         return error, True


    def decode(self, syndrome: np.ndarray[int], debug: bool = False) -> tuple[np.ndarray, bool]:
        """
        Decode the syndrome using an ordered statistics decoder (with PLU decomposition).

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, pre_proccessing_success, posterior), default is False.

        Returns
        -------
        error : nd.array
            The predicted error mechanism.
        bool
            Whether the decoding was successful.
        """
        self.field._validate(syndrome)
        super().decode(syndrome)

        if self.certainties is None:
            # WARNING: Lose information here in the qudit case???
            self.certainties = np.sum(self.posterior[:, 1:], axis=1)

        n_errors = self.h.shape[1]

        # Step 1: order the errors by likelihood
        permutation, inv_permutation = self._find_permutation(self.certainties)
        self.h = self.h[:, permutation]

        # Step 2: decompose h_eff into rank(h_eff) linearly independent columns and rows (P)
        h_rref, syndrome_rref, pivot_cols, pivot_rows, pivots = self.field.rref(self.h, syndrome)
        P = self.h[:, pivot_cols][pivot_rows, :]

        # Step 3: solve (wrt order) for the error mechanism with highest likelihood
        if self.order == 0:
            # Solve linear system P * short_error = syndrome (from rref) -> h_eff * error = syndrome with 0s to extend short_error
            error = np.zeros(n_errors, dtype=int)
            ind = [pivot_rows.index(i) for i in sorted(pivot_rows)]
            error[np.array(pivot_cols)[ind]] = syndrome_rref

            assert ((self.h @ error) % self.field.p == syndrome).all()

            error = error[inv_permutation]
        else:
            raise NotImplementedError("OSD with order > 0 is not implemented yet.")

        # Invert permutation
        self.h = self.h[:, inv_permutation]

        assert ((self.h @ error) % self.field.p == syndrome).all()

        if debug:
            return error, True, False, self.posterior
        else:
            return error, True


class DOSD(Decoder):
    """Decoder combing Dijkstra and Ordered Statistics Decoder."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], prior: np.ndarray, order: int = 0):
        """
        Initialise a D+OSD decoder.
        
        Parameters
        ----------
        posterior : nd.array
            The posterior probabilities of each error mechanism.
        order : int
            The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
        """
        super().__init__(field, h, error_channel)
        self.prior = prior
        self.order = order

    def decode(self, syndrome: np.ndarray, debug: bool = False) -> tuple[np.ndarray, bool]:
        """
        Decode the syndrome using D+OSD (Dijkstra and Ordered Statistics Decoder).

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, d_success, posteriors), default is False.

        Returns
        -------
        error : nd.array
            The predicted error mechanism.
        bool
            Whether the decoding was successful.
        """
        dijkstra = Dijkstra(self.field, self.h, self.error_channel)
        certainties, _ = -dijkstra.decode(self.h, syndrome)  # negative for ordering: low distance = high likelihood

        osd = OSD(self.field, self.h, self.error_channel, self.prior, certainties, self.order)
        return osd.decode(syndrome, debug)


class BPOSD(Decoder):
    """Decoder combining Belief Propagation and Ordered Statistics Decoder."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 1000, order: int = 0):
        """
        Initialise a BP+OSD decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations for belief propagation, default is 1000.
        order : int
            The order of the OSD algorithm, i.e. the number of dependent error mechanisms to consider. Default is 0.
        """
        super().__init__(field, h, error_channel)
        self.max_iter = max_iter
        self.order = order


    def decode(self, syndrome: np.ndarray[int], debug: bool = False) -> tuple[np.ndarray, bool]:
        """
        Decode the syndrome using BP+OSD (Belief Propagation and Ordered Statistics Decoder).

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False.

        Returns
        -------
        error : nd.array
            The predicted error mechanism.
        bool
            Whether the decoding was successful.
        """
        bp = BP(self.field, self.h, self.error_channel, self.max_iter)

        error, success, bp_success, posterior = bp.decode(syndrome, debug=True)
        if success:
            if debug:
                return error, success, bp_success, posterior
            else:
                return error, success

        # Use sum of all likelihoods of X^k/Z^k errors on a  given qudit to rank h_eff columns
        # WARNING: Lose information here in the qudit case???
        certainties = np.sum(np.delete(posterior, 0, axis=1), axis=1)
        osd = OSD(self.field, self.h, self.error_channel, posterior, certainties, self.order)
        return osd.decode(syndrome, debug)


# TODO: Generate prior in advance in simulation, to be used in all shots

# TODO: Awkward terminology, error_channel is redundant for decoders calling it prior and posterior
