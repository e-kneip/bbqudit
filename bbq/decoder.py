"""Implementation of a selection of decoders for qudits."""

from bbq.utils import err_to_det, det_to_err, norder
from bbq.field import Field

from abc import ABC, abstractmethod
from numba import njit
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
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
                "error_channel must have the same number of rows as there are error mechanisms, i.e. columns of h, and same number of columns as the field size."
            )
        # if not np.all(0 <= error_channel) & np.all(error_channel <= 1) & (np.isclose(np.sum(error_channel), 1) | np.all(np.isclose(np.sum(error_channel, axis=1), 1))):  # TODO: not sure when first of or statements happens...
        #     raise ValueError("error_channel must be filled with probabilities, which sum to 1")
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
            Q_perm = Q[errs[:, 0], i, :].copy()
            Q_perm = np.take_along_axis(
                Q_perm, self.permutation[errs[:, 1], :], axis=1
            )
            # Q_perm = self.rearange_Q(Q_perm, errs, i, permutation) (NOTE: older code, is about the same speed as the above line but maybe slower for larger simulations?)

            # Fourier transform the relevant error messages
            convolution = np.fft.fft(Q_perm, axis=1)

            # Compute the product of the probabilities for the error messages, excluding one row of messages to avoid feedback
            conv_i = np.where(convolution == 0)[0]  # todo: should be using isclose??

            if len(conv_i) == 0:
                sub_convolutions = np.prod(convolution, axis=0)
                sub_convolutions = sub_convolutions / convolution
            else:  # Not sure if this ever happens (tried a few 1000 trials), but if there ever is a 0 in conolution, will need this!
                sub_convolutions = np.empty_like(convolution)
                mask = np.ones_like(convolution, dtype=bool)
                for j in range(convolution.shape[0]):
                    mask[j] = False
                    sub_convolutions[j] = np.prod(convolution, axis=0, where=mask)
                    mask[j] = True

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

            # Prevent /0 in parallelisation
            post_i = np.where(posterior == 0)[0]  # todo: should be using isclose??

            if len(post_i) == 0:
                sub_posteriors = np.prod(posterior, axis=0) * self.prior[i, :]
                sub_posteriors = sub_posteriors / posterior
            else:
                sub_posteriors = np.empty_like(posterior)
                mask = np.ones_like(posterior, dtype=bool)
                for j in range(posterior.shape[0]):
                    mask[j] = False
                    sub_posteriors[j] = np.prod(posterior, axis=0, where=mask) * self.prior[j, :]
                    mask[j] = True

            # Pass normalised messages
            Q[i, dets[:, 0], :] = (
                sub_posteriors / np.sum(sub_posteriors, axis=1)[:, np.newaxis]
            )


    def _calculate_posterior(self, P):
        """Calculate the posterior probabilities and make hard decision on error."""
         # For errors which do not flag any detectors, use original prior
        posteriors = self.prior.copy()

        errs = list(self.err_neighbourhood.keys())
        posteriors[errs, :] = (
            np.array([np.prod(P[self.err_neighbourhood[i][:, 0], i, :], axis=0) for i in errs])
            * self.prior[errs, :]
        )

        # I think this is the same as above??? Is one of them faster??? Why did I keep both?!
        # for i, dets in self.err_neighbourhood.items():
        #     # TODO: Vectorize this:
        #     posterior = np.prod(P[dets[:, 0], i, :], axis=0) * self.prior[i, :]
        #     posteriors[i, :] = posterior

        # Using probabilities
        posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]
        max_prob = np.argmax(posteriors, axis=1)
        error = np.array(
            [
                max_prob[i] if posteriors[i, max_prob[i]] >= 0.5 else 0
                for i in range(posteriors.shape[0])
            ]
        )

        # Using likelihoods
        # posteriors /= (
        #     np.sum(posteriors, axis=1)[:, np.newaxis] - posteriors
        # )  ####### do I have blowing up problems here??? yes, yes you do...

        # max_lik = np.argmax(posteriors, axis=1)
        # # if 50:50 chance between 2 errors, max_lik will pick the 1st in row (lower power)
        # error = np.array(
        #     [
        #         max_lik[i] if posteriors[i, max_lik[i]] >= 1 else 0
        #         for i in range(posteriors.shape[0])
        #     ]
        # )

        return error, posteriors


    def decode(self, syndrome: np.ndarray, debug: bool = False, metric: bool = False) -> tuple[np.ndarray[int], bool]:
        """Decode the syndrome using belief propagation.

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False. Use if post-processing results.
        metric : bool
            Whether to return posteriors calculated at each iteration.

        Returns
        -------
        error : nd.array
            The predicted error.
        success : bool
            Whether the decoding converged to a valid solution.
        """
        if not isinstance(syndrome, np.ndarray):
            raise TypeError("syndrome must be a numpy array")
        if metric:
            posterior_track = [self.prior.tolist()]

        for _ in range(self.max_iter):
            Pi, Pj, Pk = np.where(self.P == np.inf)

            # Pass messages
            self._check_to_error_message(syndrome, self.P, self.Q)
            self._error_to_check_message(self.P, self.Q)
            # TODO: should be doing err to check and check to err in one iter not the other way around!

            self.P[Pi, Pj, Pk] = np.inf * np.ones_like(self.P[Pi, Pj, Pk])

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(self.P)

            if metric:
                posterior_track.append(posteriors.tolist())

            # Check convergence
            if np.all(self.h @ error % self.field.p == syndrome):
                if metric:
                    return error, True, True, posteriors, posterior_track
                if debug:
                    return error, True, True, posteriors
                else:
                    return error, True

        if metric:
            return error, False, False, posteriors, posterior_track
        if debug:
            return error, False, False, posteriors
        else:
            return error, False


class BPM(BP):
    """Decoder using averages to cut BP symmetry."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 1000, reset: int = 12, threshold: float = 0.001):
        """Initialise a BPM decoder.
        
        Parameters
        ----------
        reset : int
            After how many iterations to reset the node with greatest oscillations, default is 12.
        threshold : float
            Stop BP iterations if oscillations are within threshold, default is 0.001.
        """
        super().__init__(field, h, error_channel, max_iter)
        self.reset = reset
        self.threshold = threshold

    def decode(self, syndrome: np.ndarray, debug: bool = False, metric: bool = False) -> tuple[np.ndarray[int], bool]:
        if not isinstance(syndrome, np.ndarray):
            raise TypeError("syndrome must be a numpy array")
        if metric:
            posterior_track = [self.prior.tolist()]

        mean = np.zeros((self.prior.shape[0], self.prior.shape[1], self.reset), dtype=float)

        for it in range(self.max_iter):
            Pi, Pj, Pk = np.where(self.P == np.inf)

            # Pass messages
            self._check_to_error_message(syndrome, self.P, self.Q)
            self._error_to_check_message(self.P, self.Q)
            # TODO: should be doing err to check and check to err in one iter not the other way around!

            self.P[Pi, Pj, Pk] = np.inf * np.ones_like(self.P[Pi, Pj, Pk])

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(self.P)

            if metric:
                posterior_track.append(posteriors.tolist())

            # Check convergence
            if np.all(self.h @ error % self.field.p == syndrome):
                if metric:
                    return error, True, True, posteriors, posterior_track
                if debug:
                    return error, True, True, posteriors
                else:
                    return error, True
                
            # Update most recent posteriors
            mean[:, :, it % self.reset] = posteriors

            # Every reset iterations reset posterior of one error mechanism to average
            if it % self.reset == 0:
                # Find error mechanism with largest oscillations
                ran = np.ptp(mean, axis=2)
                arg = np.argmax(ran)
                arg //= self.field.p

                # Reset the posterior for arg and send to its neighbouring detectors
                av = np.average(mean, axis=2)

                # End iterations if within threshold
                if max(ran[arg]) < self.threshold:
                    continue

            self.Q[arg, self.err_neighbourhood[arg][:, 0], :] = av[arg, :]

            # Take average, chack convergence and send to OSD
            av_posteriors = np.average(mean, axis=2)

            max_lik = np.argmax(av_posteriors, axis=1)
            error = np.array(
                [
                    max_lik[i] if av_posteriors[i, max_lik[i]] >= 1 else 0
                    for i in range(av_posteriors.shape[0])
                ]
            )
            if np.all(self.h @ error % self.field.p == syndrome):
                if metric:
                    return error, True, True, av_posteriors, posterior_track
                if debug:
                    return error, True, True, av_posteriors
                else:
                    return error, True

        if metric:
            return error, False, False, av_posteriors, posterior_track
        if debug:
            return error, False, False, av_posteriors
        else:
            return error, False


class RelayBP(BP):
    """Decoder using relay belief propagation."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 60, first_iter: int = 80, solutions: int=3, relays: int=10, mem_weight: np.ndarray=None):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations per relay leg, default is 60.
        first_iter : int
            The maximum number of iterations for the first relay leg, default is 80.
        solutions : int
            The number of RelayBP solutions to find, default is 3.
        relays : int
            The number of legs of the relay, default is 10.
        mem_weight : np.ndarray
            The memory weights for each leg (dims = variable nodes x field x relays). If None, assigns 0 for all nodes and all legs.
        """
        super().__init__(field, h, error_channel, max_iter)
        self.first_iter = first_iter
        self.solutions = solutions
        self.relays = relays
        self.mem_weight = mem_weight
        self.mem_prior = error_channel.copy()

        # Store all solutions and their likelihoods
        self.num_solutions = 0
        self.guessed_errors = np.zeros((self.solutions, self.h.shape[1]), dtype=int)
        self.solutions_lh = np.ones((self.solutions)) * -np.inf

        if mem_weight is None:
            self.mem_weight = np.zeros((self.h.shape[1], self.field.p, self.relays))

    def _error_to_check_message(self, P, Q):
        """Pass messages from errors to checks."""
        for i, dets in self.err_neighbourhood.items():
            # TODO: Vectorize this too (later) (consider using einsum)

            # Isolate the relevant check messages
            posterior = P[dets[:, 0], i, :]

            # Prevent /0 in parallelisation
            post_i = np.where(posterior == 0)[0]  # todo: should be using isclose??

            if len(post_i) == 0:
                sub_posteriors = np.prod(posterior, axis=0) * self.mem_prior[i, :]
                sub_posteriors = sub_posteriors / posterior
            else:
                sub_posteriors = np.empty_like(posterior)
                mask = np.ones_like(posterior, dtype=bool)
                for j in range(posterior.shape[0]):
                    mask[j] = False
                    sub_posteriors[j] = np.prod(posterior, axis=0, where=mask) * self.mem_prior[j, :]
                    mask[j] = True

            # Pass normalised messages
            Q[i, dets[:, 0], :] = (
                sub_posteriors / np.sum(sub_posteriors, axis=1)[:, np.newaxis]
            )

    def _calculate_posterior(self, P):
        """Calculate the posterior probabilities and make hard decision on error."""
         # For errors which do not flag any detectors, use original prior
        posteriors = self.mem_prior.copy()

        errs = list(self.err_neighbourhood.keys())
        posteriors[errs, :] = (
            np.array([np.prod(P[self.err_neighbourhood[i][:, 0], i, :], axis=0) for i in errs])
            * self.mem_prior[errs, :]
        )

        # I think this is the same as above??? Is one of them faster??? Why did I keep both?!
        # for i, dets in self.err_neighbourhood.items():
        #     # TODO: Vectorize this:
        #     posterior = np.prod(P[dets[:, 0], i, :], axis=0) * self.prior[i, :]
        #     posteriors[i, :] = posterior

        # Using probabilities
        posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]
        max_prob = np.argmax(posteriors, axis=1)
        error = np.array(
            [
                max_prob[i] if posteriors[i, max_prob[i]] >= 0.5 else 0
                for i in range(posteriors.shape[0])
            ]
        )

        # Using likelihoods
        # posteriors /= (
        #     np.sum(posteriors, axis=1)[:, np.newaxis] - posteriors
        # )  ####### do I have blowing up problems here??? yes, yes you do...

        # max_lik = np.argmax(posteriors, axis=1)
        # # if 50:50 chance between 2 errors, max_lik will pick the 1st in row (lower power)
        # error = np.array(
        #     [
        #         max_lik[i] if posteriors[i, max_lik[i]] >= 1 else 0
        #         for i in range(posteriors.shape[0])
        #     ]
        # )

        return error, posteriors

    def update_memory(self, posteriors: np.ndarray[float], leg: int):
        """Update memory prior for next leg."""
        self.mem_prior = (1 - self.mem_weight[:, :, leg]) * self.prior + self.mem_weight[:, :, leg] * posteriors

    def relay_leg(self, posterior: np.ndarray[float], syndrome: np.ndarray[int], leg: int, metric: bool = False) -> tuple[np.ndarray[int], bool, np.ndarray[float]]:
        """Run BP for one leg with the given posterior.
        
        Parameters
        ----------
        posterior : np.ndarray[float]
            The posterior to initalise the leg with.
        syndrome : np.ndarray[int]
            The syndrome to correct.
        metric : bool
            Whether to keep track of posteriors.
            
        Returns
        -------
        error : np.ndarray[int]
            The decoded error.
        success : bool
            Whether BP converged to a valid error.
        posterior : np.ndarray[float]
            The final posteriors.
        """
        if metric:
            posterior_track = []

        # TODO: max_iter should be larger for first leg

        # Initialise prior and Q (error-to-check message) with previous leg
        self.mem_prior = posterior.copy()
        for i in range(self.h.shape[1]):
            # Send the same message of priors for each error to its neighbouring detectors
            if i in self.err_neighbourhood:
                self.Q[i, self.err_neighbourhood[i][:, 0], :] = self.mem_prior[i]

        if leg == 0:
            it = self.first_iter
        else:
            it = self.max_iter

        for _ in range(it):

            Pi, Pj, Pk = np.where(self.P == np.inf)

            # Pass messages
            self._check_to_error_message(syndrome, self.P, self.Q)
            self._error_to_check_message(self.P, self.Q)
            # TODO: should be doing err to check and check to err in one iter not the other way around!

            self.P[Pi, Pj, Pk] = np.inf * np.ones_like(self.P[Pi, Pj, Pk])

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(self.P)

            if metric:
                posterior_track.append(posteriors.tolist())

            # Check convergence
            if np.all(self.h @ error % self.field.p == syndrome) and not np.all(error == self.guessed_errors[self.num_solutions-1]):
                self.guessed_errors[self.num_solutions, :] = error
                self.solutions_lh[self.num_solutions] = self.score(error)
                self.num_solutions += 1
                
            # Update memory prior
            self.update_memory(posteriors, leg)

            if self.num_solutions >= self.solutions:
                break

        if metric:
            return posteriors, posterior_track
        else:
            return posteriors

    def score(self, error: np.ndarray[int]) -> int:
        """Score the error based on the prior.
        
        Parameters
        ----------
        error : np.ndarray[int]
            The error to score.
            
        Returns
        -------
        score : float
            The likelihood of the error.
        """
        score = 0

        for i, err in enumerate(error):
            post = self.prior[i, err]
            if post > 0:
                score += np.log(post)
            else:
                score -= 1000

        return score

    def decode(self, syndrome: np.ndarray, debug: bool = False, metric: bool = False) -> tuple[np.ndarray[int], bool]:
        """Decode the syndrome using belief propagation.

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, posteriors), default is False. Use if post-processing results.
        metric : bool
            Whether to return posteriors calculated at each iteration.

        Returns
        -------
        error : nd.array
            The predicted error.
        success : bool
            Whether the decoding converged to a valid solution.
        """
        posterior = self.prior.copy()
        if metric:
            posterior_track = [self.prior.tolist()]
        
        for leg in range(self.relays):
            # Run one leg
            if metric:
                posterior, leg_posteriors = self.relay_leg(posterior, syndrome, leg, metric=True)
                posterior_track.extend(leg_posteriors)
            else:
                posterior = self.relay_leg(posterior, syndrome, leg)

            if self.num_solutions >= self.solutions:
                break
        
        if not self.num_solutions:
            error = np.zeros(self.h.shape[1])
            if metric:
                return error, False, posterior_track
            if debug:
                return error, False, posterior
            return error, False

        final_error = self.guessed_errors[np.argmax(self.solutions_lh), :]
        if metric:
            return final_error, True, posterior_track
        elif debug:
            return final_error, True, posterior
        return final_error, True


class ProdRelayBP(RelayBP):
    """RelayBP decoder with a weighted product rather than a weighted sum in error-to-check message update."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 60, first_iter: int = 80, solutions: int=3, relays: int=10, mem_weight: np.ndarray=None):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations per relay leg, default is 60.
        first_iter : int
            The maximum number of iterations for the first relay leg, default is 80.
        solutions : int
            The number of RelayBP solutions to find, default is 3.
        relays : int
            The number of legs of the relay, default is 10.
        mem_weight : np.ndarray
            The memory weights for each leg (dims = variable nodes x field x relays). If None, assigns 0 for all nodes and all legs.
        """
        super().__init__(field, h, error_channel, max_iter, first_iter, solutions, relays, mem_weight)
        self.qmem_prior = np.ones_like(self.mem_prior)

    def update_memory(self, posteriors: np.ndarray[float], leg: int):
        """Update memory prior for next leg."""
        self.qmem_prior = self.qmem_prior**self.mem_weight[:, :, leg] * posteriors
        self.mem_prior = self.prior * self.qmem_prior


class SmoothRelayBP(RelayBP):
    """RelayBP decoder with a memory scaled wrt loop size."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 60, first_iter: int = 80, solutions: int=3, relays: int=10, mem_weight: np.ndarray=None, loop_size: int=1):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations per relay leg, default is 60.
        first_iter : int
            The maximum number of iterations for the first relay leg, default is 80.
        solutions : int
            The number of RelayBP solutions to find, default is 3.
        relays : int
            The number of legs of the relay, default is 10.
        mem_weight : np.ndarray
            The memory weights for each leg (dims = variable nodes x field x relays). If None, assigns 0 for all nodes and all legs.
        loop_size : int
            The size of the smallest loop in the Tanner graph.
            TODO: Make this dependent on the detector (doesn't matter if translationally invariant like BB codes)
        """
        super().__init__(field, h, error_channel, max_iter, first_iter, solutions, relays, mem_weight)
        if not loop_size > 0:
            raise ValueError(f"loop_size must be a positive integer, not {loop_size}.")
        self.loop_size = loop_size
        self.qmem_prior = np.zeros_like(self.mem_prior)

    def update_memory(self, posteriors: np.ndarray[float], leg: int, iteration: int):
        """Update memory prior for next leg."""
        self.mem_prior = self.prior + self.qmem_prior + posteriors
        if (iteration % self.loop_size == 0):
            self.qmem_prior *= self.mem_weight[:, :, leg]
        self.qmem_prior += (posteriors * self.mem_weight[:, :, leg])

    def relay_leg(self, posterior: np.ndarray[float], syndrome: np.ndarray[int], leg: int, metric: bool = False) -> tuple[np.ndarray[int], bool, np.ndarray[float]]:
        """Run BP for one leg with the given posterior.
        
        Parameters
        ----------
        posterior : np.ndarray[float]
            The posterior to initalise the leg with.
        syndrome : np.ndarray[int]
            The syndrome to correct.
        metric : bool
            Whether to keep track of posteriors.
            
        Returns
        -------
        error : np.ndarray[int]
            The decoded error.
        success : bool
            Whether BP converged to a valid error.
        posterior : np.ndarray[float]
            The final posteriors.
        """
        if metric:
            posterior_track = []

        # TODO: max_iter should be larger for first leg

        # Initialise prior and Q (error-to-check message) with previous leg
        self.mem_prior = posterior.copy()
        for i in range(self.h.shape[1]):
            # Send the same message of priors for each error to its neighbouring detectors
            if i in self.err_neighbourhood:
                self.Q[i, self.err_neighbourhood[i][:, 0], :] = self.mem_prior[i]

        if leg == 0:
            it = self.first_iter
        else:
            it = self.max_iter

        for iteration in range(it):

            Pi, Pj, Pk = np.where(self.P == np.inf)

            # Pass messages
            self._check_to_error_message(syndrome, self.P, self.Q)
            self._error_to_check_message(self.P, self.Q)
            # TODO: should be doing err to check and check to err in one iter not the other way around!

            self.P[Pi, Pj, Pk] = np.inf * np.ones_like(self.P[Pi, Pj, Pk])

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(self.P)

            if metric:
                posterior_track.append(posteriors.tolist())

            # Check convergence

            ##########################
            # this is broken bcos I changed relaybp to store solutions in self.guessed_error so number of outputs of leg is wrong!!!!
            ##########################

            if np.all(self.h @ error % self.field.p == syndrome):
                if metric:
                    return error, True, posteriors, posterior_track
                else:
                    return error, True, posteriors

            # Update memory prior
            self.update_memory(posteriors, leg, iteration)

        if metric:
            return error, False, posteriors, posterior_track
        else:
            return error, False, posteriors


class DoubleRelayBP(SmoothRelayBP):
    """RelayBP decoder with double memory: within loops and betwen loops."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 60, first_iter: int = 80, solutions: int=3, relays: int=10, mem_weight: np.ndarray=None, micro_mem_weight: np.ndarray=None, loop_size: int=1):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations per relay leg, default is 60.
        first_iter : int
            The maximum number of iterations for the first relay leg, default is 80.
        solutions : int
            The number of RelayBP solutions to find, default is 3.
        relays : int
            The number of legs of the relay, default is 10.
        mem_weight : np.ndarray
            The memory weights for each leg (dims = variable nodes x field x relays). If None, assigns 0 for all nodes and all legs.
        loop_size : int
            The size of the smallest loop in the Tanner graph.
            TODO: Make this dependent on the detector (doesn't matter if translationally invariant like BB codes)
        """
        super().__init__(field, h, error_channel, max_iter, first_iter, solutions, relays, mem_weight, loop_size)
        self.micro_mem_weight = micro_mem_weight
        if micro_mem_weight is None:
            self.micro_mem_weight = np.zeros((self.h.shape[1], self.field.p, self.relays))
        self.qmicro_mem_prior = np.zeros_like(self.mem_prior)

    def update_memory(self, posteriors: np.ndarray[float], leg: int, iteration: int):
        """Update memory prior for next leg."""
        if iteration % self.loop_size == 0:
            self.qmem_prior += self.qmicro_mem_prior
            self.qmem_prior *= self.mem_weight[:, :, leg]
            self.qmicro_mem_prior = np.zeros_like(self.mem_prior)

        self.qmicro_mem_prior *= self.micro_mem_weight[:, :, leg]
        self.qmicro_mem_prior += posteriors

        self.mem_prior = self.prior + self.qmem_prior + self.qmicro_mem_prior


class StrawBP(RelayBP):
    """Decoder using alternating legs of MemBP and standard BP as an ensemble."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter_bp: int = 60, max_iter_mem: int = 40, solutions: int=3, relays: int=3, runners: int=2, centre: float=0.05, width: float=0.2):
        """Initialise a belief propagation decoder.
        
        Parameters
        ----------
        max_iter_bp : int
            The maximum number of iterations per BP stem, default is 60.
        max_iter_mem : int
            The maximum number of iterations per MemBP runner, default is 40.
        solutions : int
            The number of StrawBP solutions to find, default is 3.
        relays : int
            The number of layers of StrawBP to run (one layer = MemBP runner + BP stem), default is 3.
        runners : int
            The number of MemBP runners to create in every layer, default is 2.
        centre : float
            The centre of the normal distribution of coefficients in MemBP runners.
        width : float
            The width of the normal distribution of coefficients in MemBP runners.
        """
        # Use of iters in RelayBP overwritten and mem_weight found explicitly
        super().__init__(field, h, error_channel, 10, 10, solutions, relays, None)
        self.max_iter_bp = max_iter_bp
        self.max_iter_mem = max_iter_mem
        self.runners = runners
        self.centre = centre
        self.width = width

    def update_memory(self, posteriors: np.ndarray[float]):
        """Update memory prior for next leg."""
        self.mem_prior = (1 - self.mem_weight) * self.prior + self.mem_weight * posteriors

    def relay_leg(self, posterior: np.ndarray[float], syndrome: np.ndarray[int], no_mem = False, metric: bool = False) -> tuple[np.ndarray[int], bool, np.ndarray[float]]:
        """Run BP for one leg with the given posterior.
        
        Parameters
        ----------
        posterior : np.ndarray[float]
            The posterior to initalise the leg with.
        syndrome : np.ndarray[int]
            The syndrome to correct.
        no_mem : bool
            Use prior instead of mem_prior (i.e. revert to standard BP)
        metric : bool
            Whether to keep track of posteriors.
            
        Returns
        -------
        error : np.ndarray[int]
            The decoded error.
        success : bool
            Whether BP converged to a valid error.
        posterior : np.ndarray[float]
            The final posteriors.
        """
        if metric:
            posterior_track = []

        # TODO: max_iter should be larger for first leg

        # Initialise prior and Q (error-to-check message) with previous leg
        self.mem_prior = posterior.copy()
        for i in range(self.h.shape[1]):
            # Send the same message of priors for each error to its neighbouring detectors
            if i in self.err_neighbourhood:
                self.Q[i, self.err_neighbourhood[i][:, 0], :] = self.mem_prior[i]

        if no_mem:
            its = self.max_iter_bp
        else:
            its = self.max_iter_mem

        for _ in range(its):
            Pi, Pj, Pk = np.where(self.P == np.inf)

            # Pass messages
            self._check_to_error_message(syndrome, self.P, self.Q)
            self._error_to_check_message(self.P, self.Q)
            # TODO: should be doing err to check and check to err in one iter not the other way around!

            self.P[Pi, Pj, Pk] = np.inf * np.ones_like(self.P[Pi, Pj, Pk])

            # Calculate posterior and make hard decision on errors
            error, posteriors = self._calculate_posterior(self.P)

            if metric:
                posterior_track.append(posteriors.tolist())

            # Check convergence
            if np.all(self.h @ error % self.field.p == syndrome) and not np.all(error == self.guessed_errors[self.num_solutions-1]):
                self.guessed_errors[self.num_solutions, :] = error
                self.solutions_lh[self.num_solutions] = self.score(error)
                self.num_solutions += 1
                
            # Update memory prior
            if not no_mem:
                self.update_memory(posteriors)

            if self.num_solutions >= self.solutions:
                break

        if metric:
            return posteriors, posterior_track
        else:
            return posteriors

    def decode(self, syndrome: np.ndarray, debug: bool = False, metric: bool = False) -> tuple[np.ndarray[int], bool]:
        """Decode the syndrome using belief propagation.

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False. Use if post-processing results.
        metric : bool
            Whether to return posteriors calculated at each iteration.

        Returns
        -------
        error : nd.array
            The predicted error.
        success : bool
            Whether the decoding converged to a valid solution.
        """
        posterior = self.prior.copy()
        if metric:
            posterior_track = [self.prior.tolist()]

        ######### need way to keep track of which strawberries have grown and skip their for loop!!!
        
        for stem in range(self.relays):
            # Run MemBP
            # Set memory strengths
            self.mem_weight = np.random.uniform(self.centre - self.width/2, self.centre + self.width/2, size=(self.h.shape[1], self.field.p))

            if metric:
                posterior, leg_posteriors = self.relay_leg(posterior, syndrome, no_mem=False, metric=True)
                posterior_track.extend(leg_posteriors)
            else:
                posterior = self.relay_leg(posterior, syndrome, no_mem=False)

            if self.num_solutions >= self.solutions:
                break

            # Run BP
            if metric:
                posterior, leg_posteriors = self.relay_leg(posterior, syndrome, no_mem=True, metric=True)
                posterior_track.extend(leg_posteriors)
            else:
                posterior = self.relay_leg(posterior, syndrome, no_mem=True)

            if self.num_solutions >= self.solutions:
                break
        
        if not self.num_solutions:
            error = np.zeros(self.h.shape[1])
            if metric:
                return error, False, posterior_track
            if debug:
                return error, False, posterior
            return error, False

        final_error = self.guessed_errors[np.argmax(self.solutions_lh), :]
        if metric:
            return final_error, True, posterior_track
        elif debug:
            return final_error, True, posterior
        return final_error, True


class OSD(Decoder):
    """Decoder using ordered statistics decoding."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], posterior: np.ndarray[float], certainties: np.ndarray[float] | None = None, order: int = 0, power: int = 1, norder: np.ndarray[int] = 0):
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
        power : int
            The number of most likely error powers to consider for each error mechanism. Default is 1.
        norder : nd.array[int]
            Precomputed norder masks to use for the given order, default 0 computes them internally.
        """
        if not order >= 0:
            raise ValueError("order must be a non-negative integer")
        if power > 1:
            raise NotImplementedError("power > 1 is not implemented yet. Big linear combos to figures out...")
        
        super().__init__(field, h, error_channel)
        self.posterior = posterior
        self.certainties = certainties
        self.order = order
        if power > 1:
            raise NotImplementedError("power > 1 is not implemented yet.")
        self.power = power

        if order:
            if np.any(norder == 1):
                self.order_mask = norder
            else:
                _, _, pivot_cols, pivot_rows, _ = self.field.rref(self.h, np.zeros(self.h.shape[0], dtype=int))
                rank = min(len(pivot_cols), len(pivot_rows))  # could also do n - k = m = 2*rank, or store the output of rref for later
                dim = self.h.shape[1] - rank
                if order == 1:
                    self.order_mask = self._order_one(dim)
                else:
                    self.order_mask = norder(dim, self.order)
            self.power_like = self._power()
    
    def _order_one(self, dim):
        """Generate all binary masks of errors with up to 1 non-zero entry."""
        masks = np.eye(dim, dtype=int)
        return np.vstack((np.zeros((1, dim), dtype=int), masks))

    def _power(self):
        """Order the non-0 powers of error mechanisms by likelihood, i.e. for error mechanism i, cover[i] gives the indices of the error powers in descending order of likelihood."""
        cover = np.argsort(-self.posterior[:, 1:], axis = 1)
        return cover + 1

    def _find_permutation(self, certainties):
        """Find the permutation of the error mechanisms based on their likelihood."""

        permutation = np.argsort(-certainties, axis=None)  # high certainty = low index
        inv_permutation = np.empty_like(permutation)
        inv_permutation[permutation] = np.arange(len(permutation))

        return permutation, inv_permutation

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

        n_detectors, n_errors = self.h.shape

        # Step 1: order the errors by likelihood
        permutation, inv_permutation = self._find_permutation(self.certainties)
        self.h = self.h[:, permutation]
        if self.order:
            self.posterior = self.posterior[permutation, :]

        # Step 2: decompose h_eff into rank(h_eff) linearly independent columns and rows (P) and the remainder (B)
        h_rref, syndrome_rref, pivot_cols, pivot_rows, pivots = self.field.rref(self.h, syndrome)
        if self.order:
            P = h_rref[:, pivot_cols]  # P is square identity matrix
            B = h_rref[:, [i for i in range(n_errors) if i not in pivot_cols]]

        # Step 3: solve (wrt order) for the error mechanism with highest likelihood
        if not self.order:
            # Solve linear system P * short_error = syndrome (from rref) -> h_eff * error = syndrome with 0s to extend short_error
            error = np.zeros(n_errors, dtype=int)
            ind = [pivot_rows.index(i) for i in sorted(pivot_rows)]
            error[np.array(pivot_cols)[ind]] = syndrome_rref

            assert ((self.h @ error) % self.field.p == syndrome).all()

            error = error[inv_permutation]
        elif self.order:
            winning_error, winning_score = np.zeros(n_errors, dtype=int), -np.inf

            # All possible combos of error mechanisms on non-pivot columns of size 'order'
            dim = B.shape[1]
            guess = np.zeros(dim, dtype=int)
            ind = [pivot_rows.index(i) for i in sorted(pivot_rows)]
            self.power_like = self.power_like[permutation, :]
            self.power_like = self.power_like[P.shape[1]:, :]
            guesses = self.order_mask * self.power_like[:, self.power - 1]  # only works for order = 1 or power = 1, o/w need to do lin combos
            for guess in guesses:
                # Solve linear system P * short_error = syndrome - B * guess
                error = np.zeros(n_errors, dtype=int)
                remainder = (syndrome_rref - B @ guess) % self.field.p
                error[np.array(pivot_cols)[ind]] = remainder
                error[[i for i in range(n_errors) if i not in pivot_cols]] = guess

                assert ((self.h @ error) % self.field.p == syndrome).all()

                # Score the guess
                score = 0
                for i, err in enumerate(error):
                    post = self.posterior[i, err]
                    if post > 0:
                        score += np.log(post)
                    else:
                        score -= 1000
                if score > winning_score:
                    winning_error, winning_score = error[inv_permutation], score
            error = winning_error


        # Invert permutation
        self.h = self.h[:, inv_permutation]
        if self.order:
            self.posterior = self.posterior[inv_permutation, :]

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


    def decode(self, syndrome: np.ndarray[int], debug: bool = False, metric: bool = False) -> tuple[np.ndarray, bool]:  # TODO: do I wanna keep metric? and if so currently should have metric OR debug enabled and metric is kinda a superset of debug so can be cleaner!
        """
        Decode the syndrome using BP+OSD (Belief Propagation and Ordered Statistics Decoder).

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False.
        metric : bool
            Whether to return the posteriors calculated at each oteration of BP.

        Returns
        -------
        error : nd.array
            The predicted error mechanism.
        bool
            Whether the decoding was successful.
        """
        bp = BP(self.field, self.h, self.error_channel, self.max_iter)

        if metric:
            error, success, bp_success, posterior, posterior_track = bp.decode(syndrome, metric=True)
            if success:
                return error, success, bp_success, posterior_track
        else:
            error, success, bp_success, posterior = bp.decode(syndrome, debug=True)
            if success:
                if debug:
                    return error, success, bp_success, posterior
                else:
                    return error, success

        # Use sum of all likelihoods of X^k/Z^k errors on a  given qudit to rank h_eff columns
        # WARNING: Lose information here in the qudit case???
        certainties = np.max(np.delete(posterior, 0, axis=1), axis=1)
        osd = OSD(self.field, self.h, self.error_channel, posterior, certainties, self.order)
        if metric:
            error, success = osd.decode(syndrome)
            return error, success, False, posterior_track
        else:
            return osd.decode(syndrome, debug)

class BPMOSD(BPOSD):
    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 1000, order: int = 0, reset: int = 12, threshold: float = 0.001):
        super().__init__(field, h, error_channel, max_iter, order)
        self.reset = reset
        self.threshold = threshold

    def decode(self, syndrome: np.ndarray[int], debug: bool = False, metric: bool = False) -> tuple[np.ndarray, bool]:  # TODO: do I wanna keep metric? and if so currently should have metric OR debug enabled and metric is kinda a superset of debug so can be cleaner!
        """
        Decode the syndrome using BP+OSD (Belief Propagation and Ordered Statistics Decoder).

        Parameters
        ----------
        syndrome : nd.array
            The syndrome of the error.
        debug : bool
            Whether to return debug information (error, success, bp_success, posteriors), default is False.
        metric : bool
            Whether to return the posteriors calculated at each oteration of BP.

        Returns
        -------
        error : nd.array
            The predicted error mechanism.
        bool
            Whether the decoding was successful.
        """
        bpm = BPM(self.field, self.h, self.error_channel, self.max_iter, self.reset, self.threshold)

        if metric:
            error, success, bp_success, posterior, posterior_track = bpm.decode(syndrome, metric=True)
            if success:
                return error, success, bp_success, posterior_track
        else:
            error, success, bp_success, posterior = bpm.decode(syndrome, debug=True)
            if success:
                if debug:
                    return error, success, bp_success, posterior
                else:
                    return error, success

        # Use sum of all likelihoods of X^k/Z^k errors on a  given qudit to rank h_eff columns
        # WARNING: Lose information here in the qudit case???
        certainties = np.max(np.delete(posterior, 0, axis=1), axis=1)
        osd = OSD(self.field, self.h, self.error_channel, posterior, certainties, self.order)
        if metric:
            error, success = osd.decode(syndrome)
            return error, success, False, posterior_track
        else:
            return osd.decode(syndrome, debug)


class BPLSDbin(Decoder):
    """Decode using BP on qudits then binarise for LSD."""
    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], max_iter: int = 1000, bits_per_step: int = None, lsd_order: int = 0, lsd_method: str = 'LSD_0'):
        """
        Initialise a BP+LSDbin decoder.
        
        Parameters
        ----------
        max_iter : int
            The maximum number of iterations for belief propagation, default is 1000.
        bits_per_step : int
            Specifies the number of bits added to the cluster in each step of the LSD algorithm. If none given, set to block length of code.
        lsd_order : int
            The order of the LSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
        lsd_method : str
            The LSD method of the LSD algorithm applied to each cluster. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}. By default 'LSD_0'.
        """
        super().__init__(field, h, error_channel)
        self.max_iter = max_iter
        self.lsd_order = lsd_order
        self.lsd_method = lsd_method

        if bits_per_step == None:
            self.bits_per_step = self.h.shape[1]
        else:
            self.bits_per_step = bits_per_step

    def binarise(self, array: np.ndarray[int]):
        """
        Rewrite an array (in the given field) as a binary array.

        Parameters
        ----------
        array : np.ndarray[int]
            The array to be binarised.
        
        Returns
        -------
        array_bin : np.ndarray[int]
            The binarised array.
        """
        dims = array.shape
        if not len(dims) in [1, 2]:
            raise ValueError(f"Array must be 1 or 2 dimensional, not {len(dims)}.")

        # Vector case
        if len(dims) == 1:
            array_bin = np.zeros(dims[0] * (self.field.p - 1), dtype=int)
            zeros = np.nonzero(array)[0]
            for z in zeros:
                array_bin[z * (self.field.p - 1) + array[z] - 1] = 1

        # Matrix case
        elif len(dims) == 2:
            array_bin = np.zeros((dims[0] * (self.field.p - 1), dims[1] * (self.field.p - 1)), dtype=int)
            for col in range(dims[1]):
                zeros = np.nonzero(array[:, col])[0]
                for i in range(self.field.p - 1):
                    for z in zeros:
                        array_bin[z * (self.field.p - 1) + (array[z, col] * (i+1)) % self.field.p - 1, (self.field.p - 1) * col + i] = 1

        return array_bin
    
    def ditarise(self, array_bin: np.ndarray[int]):
        """
        Rewrite an array (in binary) as a dit array.

        Parameters
        ----------
        array_bin : np.ndarray[int]
            The binary array to be converted.
        
        Returns
        -------
        array_bin : np.ndarray[int]
            The dit array.
        """
        dims = array_bin.shape
        if not len(dims) == 1:
            raise ValueError(f"Array must be 1 dimensional, not {len(dims)}.")
        
        qudits = dims[0] // (self.field.p - 1)
        array = np.zeros(qudits, dtype=int)
        zeros = np.nonzero(array_bin)[0]

        for z in zeros:
            array[z // (self.field.p - 1)] = z % (self.field.p - 1) + 1

        return array


    def decode(self, syndrome: np.ndarray[int], debug: bool = False) -> tuple[np.ndarray, bool]:
        """
        Decode the syndrome using BP+LSDbin (Belief Propagation and (binarised) Localised Statistics Decoder).

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

        # Binarise BP output for LSD
        h_bin, syndrome_bin = self.binarise(self.h), self.binarise(syndrome)
        posterior_bin = np.delete(posterior, 0, axis=1).flatten()

        # lsd = BpLsdDecoder(h_bin, error_channel=posterior_bin, max_iter=0, bits_per_step=self.bits_per_step, lsd_order=self.lsd_order, lsd_method=self.lsd_method)
        osd = BpOsdDecoder(h_bin, error_channel=list(posterior_bin), max_iter=0, osd_order=self.lsd_order, osd_method=self.lsd_method)
        error_bin = osd.decode(syndrome_bin)
        error = self.ditarise(error_bin)

        if debug:
            return error, True, False, posterior
        else:
            return error, True

# TODO: Generate prior in advance in simulation, to be used in all shots

# TODO: Awkward terminology, error_channel is redundant for decoders calling it prior and posterior
