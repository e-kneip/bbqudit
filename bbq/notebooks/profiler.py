from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.circuit import construct_sm_circuit, construct_decoding_matrix
from bbq.decoder import BPOSD, BP

import numpy as np

d = 3
field = Field(2)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 - x, 1 - y

bb = BivariateBicycle(a, b, d, d, 1)
hx, hz = bb.hx, bb.hz
lx, lz = bb.x_logicals, bb.z_logicals

x_order = ['idle', 0, 3, 1, 2]
z_order = [0, 3, 1, 2, 'idle']

num_cycles = d
circ = construct_sm_circuit(bb, x_order, z_order)

# Set up
p = 0.004

error_rates = {'Meas': p, 'Prep': p, 'idle': p, 'CNOT': p}
fails = 0

# Generate sm circuit and decoding matrix
hx_eff, short_hx_eff, hz_eff, short_hz_eff, channel_prob_x, channel_prob_z = construct_decoding_matrix(bb, circ, error_rates, num_cycles)
x_prior = np.zeros((short_hx_eff.shape[1], field.p), dtype=float)
for i, prob in enumerate(channel_prob_x):
    x_prior[i, 0] = 1 - prob
    for j in range(1, field.p):
        x_prior[i, j] = prob / (field.p - 1)

x_syndrome_history = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0])
x_error = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x_syndrome_final_logical = np.array([0, 0])

# Decode with bbq
bbq_bposd = BPOSD(field, short_hx_eff.toarray(), x_prior, max_iter=2, order=0)
bbq_bp = BP(field, short_hx_eff.toarray(), x_prior, max_iter=2)

x_error, _, _, posteriors = bbq_bp.decode(x_syndrome_history, debug=True)
# assert np.all((short_hx_eff @ x_error) % field.p == x_syndrome_history)

# Check logical effect
first_logical_row = bb.l * bb.m * (num_cycles + 2)
k = len(lx)
x_syndrome_history_augmented_guessed = (hx_eff @ x_error) % field.p
x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[first_logical_row: first_logical_row + k]

if not np.array_equal(x_syndrome_final_logical_guessed, x_syndrome_final_logical):
    bbq_success = False
else:
    bbq_success = True

print(bbq_success and np.all((short_hx_eff @ x_error) % field.p == x_syndrome_history))
