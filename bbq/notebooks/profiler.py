from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.circuit import construct_sm_circuit, generate_noisy_circuit, simulate_x_circuit
from bbq.decoder import BPOSD

import numpy as np
import pickle

with open('./circuit_level/deco_tog', 'rb') as fp:
	mats_tog = pickle.load(fp)
	
d = 5
field = Field(3)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 - x, 1 - y

bb = BivariateBicycle(a, b, d, d, 1)
hx, hz = bb.hx, bb.hz
lx, lz = bb.x_logicals, bb.z_logicals

x_order = ['idle', 0, 3, 1, 2]
z_order = [0, 3, 1, 2, 'idle']
num_cycles = d

circ = construct_sm_circuit(bb, x_order, z_order)

# Circuit-level qudit with BBQ

r, s = 0.6, 0.02
physical_error = np.array([s*(r**q) for q in range(10)])
results_sep, results_tog = [], []
error_rates = {'Meas': s, 'Prep': s, 'idle': s, 'CNOT': s}
hx_eff_tog, short_hx_eff_tog, hz_eff_tog, short_hz_eff_tog, s_channel_prob_x_tog, s_channel_prob_z_tog = mats_tog[f'{d}']

i = 6
p = physical_error[6]

error_rates = {'Meas': p, 'Prep': p, 'idle': p, 'CNOT': p}
fails_tog = 0
trials_tog = 0

# Generate sm circuit and decoding matrix
channel_prob_x_tog, channel_prob_z_tog = list((r**i)*np.array(s_channel_prob_x_tog)), list((r**i)*np.array(s_channel_prob_z_tog))
x_prior_tog = np.zeros((short_hx_eff_tog.shape[1], field.p), dtype=float)
for i, prob in enumerate(channel_prob_x_tog):
    x_prior_tog[i, 0] = 1 - prob
    for j in range(1, field.p):
        x_prior_tog[i, j] = prob / (field.p - 1)

for _ in range(5):
    # Generate noisy circ
    noisy_circ, err_cnt = generate_noisy_circuit(bb, circ * num_cycles, error_rates)
    
    # Simulate noisy circ
    x_syndrome_history, x_state, x_syndrome_map, x_err_count = simulate_x_circuit(bb, noisy_circ + circ + circ)

    # Calculate true logical effect
    qudits_dict = bb.qudits_dict
    data_qudits = bb.data_qudits
    x_state_data_qudits = [x_state[qudits_dict[qudit]] for qudit in data_qudits]
    x_syndrome_final_logical = (np.array(lz) @ x_state_data_qudits) % field.p
    
    # Syndrome sparsification
    z_checks = bb.Zchecks
    x_syndrome_history_copy = x_syndrome_history.copy()
    for check in z_checks:
        pos = x_syndrome_map[check]
        assert len(pos) == num_cycles + 2
        for row in range(1, num_cycles + 2):
            x_syndrome_history[pos[row]] += x_syndrome_history_copy[pos[row-1]]
    x_syndrome_history %= field.p

    trials_tog += 1
    # Decode
    bposd = BPOSD(field, short_hx_eff_tog.toarray(), x_prior_tog, max_iter=300)
    x_error, _ = bposd.decode(x_syndrome_history)
    assert np.all((short_hx_eff_tog @ x_error) % field.p == x_syndrome_history)
    
    # Check logical effect
    first_logical_row = bb.l * bb.m * (num_cycles + 2)
    k = len(lx)
    x_syndrome_history_augmented_guessed = (hx_eff_tog @ x_error) % field.p
    x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[first_logical_row: first_logical_row + k]
    
    if not np.array_equal(x_syndrome_final_logical_guessed, x_syndrome_final_logical):
        fails_tog += 1
results_tog.append(trials_tog)
