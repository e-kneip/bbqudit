import cProfile

import numpy as np

from bbq.bbq_code import BivariateBicycle
from bbq.decoder import bp_osd
from bbq.polynomial import Polynomial


def simulate(field, h, l, max_iter, num_failures, physical_error) -> None:
    """Run a single shot of a simulation."""
    failures = 0

    while failures < num_failures:
        # Generate syndrome
        n_qudits = h.shape[1]
        error = np.zeros(n_qudits, dtype=int)
        error_mask = np.random.rand(n_qudits) < physical_error
        for i in np.where(error_mask)[0]:
            error[i] = np.random.randint(1, field)
        syndrome = (h @ error) % field

        # Construct error probability
        channel_prob_x = np.ones(n_qudits) * physical_error

        x_prior = np.zeros((len(channel_prob_x), field), dtype=float)

        for i, prob in enumerate(channel_prob_x):
            x_prior[i, 0] = 1 - prob
            for j in range(1, field):
                x_prior[i, j] = prob / (field - 1)

        # Decode
        # guessed_error, decoder_success, bp_success, posterior = belief_propagation(field, h, syndrome, x_prior, max_iter, debug=True)
        guessed_error = bp_osd(
            field, h, syndrome, x_prior, max_iter, order=0, debug=True
        )[0]
        error_difference = (error - guessed_error) % field
        logical_effect = (np.array(l) @ error_difference) % field

        # Check success
        # if np.any(logical_effect != 0) or not decoder_success:
        if np.any(logical_effect != 0):
            failures += 1


def main():
    a = Polynomial(5, np.array([[1, 0], [-1, 0]]))
    b = Polynomial(5, np.array([[1, -1], [0, 0]]))
    bb5 = BivariateBicycle(a, b, 3, 3, 1)
    h5 = bb5.hx
    l5 = bb5.x_logicals

    physical_error = 0.07

    simulate(5, h5, l5, 1000, 5, physical_error)


if __name__ == "__main__":
    cProfile.run("main()")
