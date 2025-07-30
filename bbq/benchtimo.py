from bbq.decoder import bp_osd
import numpy as np
from bbq.polynomial import Polynomial
from bbq.bbq_code import BivariateBicycle
from datetime import datetime


def main():
    startTime = datetime.now()
    p = 0.10
    field = 3
    a = Polynomial(3, np.array([[1, 0], [-1, 0]]))
    b = Polynomial(3, np.array([[1, -1], [0, 0]]))
    bb3 = BivariateBicycle(a, b, 3, 3, 1)
    h = bb3.hx
    l = bb3.x_logicals
    max_iter = 100
    shots = 1000

    for it in range(shots):
        # Generate syndrome
        n_qudits = h.shape[1]
        error = np.zeros(n_qudits, dtype=int)
        error_mask = np.random.rand(n_qudits) < p
        for i in np.where(error_mask)[0]:
            error[i] = np.random.randint(1, field)
        syndrome = (h @ error) % field

        # Construct error probability
        channel_prob_x = np.ones(n_qudits) * p

        x_prior = np.zeros((len(channel_prob_x), field), dtype=float)

        for i, prob in enumerate(channel_prob_x):
            x_prior[i, 0] = 1 - prob
            for j in range(1, field):
                x_prior[i, j] = prob / (field - 1)

        # Decode
        # guessed_error, decoder_success, bp_success, posterior = belief_propagation(field, h, syndrome, x_prior, max_iter, debug=True)
        guessed_error, decoder_success, bp_success, posterior = bp_osd(
            field, h, syndrome, x_prior, max_iter, order=0, debug=True
        )
        error_difference = (error - guessed_error) % field
        logical_effect = (np.array(l) @ error_difference) % field
    print(f"Run time for {shots} shots: {datetime.now() - startTime}")
    print(f"Run time per shot: {(datetime.now() - startTime) / shots}")


if __name__ == "__main__":
    main()
