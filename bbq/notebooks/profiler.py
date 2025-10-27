from bbq.decoder import Decoder, BP
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.field import Field

import numpy as np
import numba
import datetime


@numba.njit
def _norder(dim, order):
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

class TestOSD(Decoder):
    """Decoder using ordered statistics decoding."""

    def __init__(self, field: Field, h: np.ndarray[int], error_channel: np.ndarray[float], posterior: np.ndarray[float], certainties: np.ndarray[float] | None = None, order: int = 0, power: int = 1):
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
            print(f'Initializing OSD of order {order} at {datetime.datetime.now()}')
            _, _, pivot_cols, pivot_rows, _ = self.field.rref(self.h, np.zeros(self.h.shape[0], dtype=int))
            rank = min(len(pivot_cols), len(pivot_rows))  # could also do n - k = m = 2*rank, or store the output of rref for later
            dim = self.h.shape[1] - rank
            print(f'Found OSD dimension at {datetime.datetime.now()}')
            if order == 1:
                self.order_mask = self._order_one(dim)
            else:
                self.order_mask = _norder(dim, self.order)
            print(f'Completed mask generation at {datetime.datetime.now()}')
            self.power_like = self._power()
            print(f'Completed power likelihood generation at {datetime.datetime.now()}')
    
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

print(f'Profiler started at {datetime.datetime.now()}')

# Set up code
field = Field(2)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 - x, 1 - y

bb = BivariateBicycle(a, b, 5, 5, 1)
hx, lx = bb.hx, bb.x_logicals
n_qudits = hx.shape[1]

max_iter = 100
physical_error = [0.1]

order = [0, 1, 2, 5]
order_results = {o: [] for o in order}
f = 1

print(f'Finished set up at {datetime.datetime.now()}')

for p in physical_error:
    fails = 0
    order_fails = {o: 0 for o in order}
    results = {o: 0 for o in order}

    # Set up BP decoder
    channel_prob_x = np.ones(n_qudits) * p
    x_prior = np.zeros((n_qudits, field.p), dtype=float)

    for i, prob in enumerate(channel_prob_x):
        x_prior[i, 0] = 1 - prob
        for j in range(1, field.p):
            x_prior[i, j] = prob / (field.p - 1)

    bp = BP(field, hx, x_prior, max_iter)

    while fails < f*len(order):
        # Generate syndrome
        error = np.zeros(n_qudits, dtype=int)
        error_mask = np.random.rand(n_qudits) < p
        for i in np.where(error_mask)[0]:
            error[i] = np.random.randint(1, field.p)
        syndrome = (hx @ error) % field.p

        # Decode with BP
        guessed_error, success, bp_success, posterior = bp.decode(syndrome, debug=True)

        # Decode with different OSD orders
        certainties = np.sum(np.delete(posterior, 0, axis=1), axis=1)
        for o in order:
            if order_fails[o] < f:
                results[o] += 1
                osd = TestOSD(field, hx, x_prior, posterior, certainties, order=o)
                guessed_error, decoder_success, bp_success, posterior = osd.decode(syndrome, debug=True)
                error_difference = (error - guessed_error) % field.p
                logical_effect = (np.array(lx) @ error_difference) % field.p

                if not (logical_effect == 0).all():
                    fails += 1
                    order_fails[o] += 1
    for o in order:
        order_results[o].append(results[o])
