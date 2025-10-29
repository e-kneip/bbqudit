from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.circuit import construct_sm_circuit, construct_decoding_matrix, generate_noisy_circuit, simulate_x_circuit

from ldpc import BpOsdDecoder

import numpy as np

print('Running circuit level simulation')
print('--------------------------------')

print('Setting up hx, lx...')
# Define parity check matrix, hx, and its logicals, lx, for the 3x3 qubit toric code
field = Field(2)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 + x, 1 + y
bb = BivariateBicycle(a, b, 3, 3, 1)
hx, lx = bb.hx, bb.x_logicals
hz, lz = bb.hz, bb.z_logicals
n_qudits = hx.shape[1]
code_name = '3x3 Qubit Toric Code'

# Define noise model parameters
x_order = ['Idle', 0, 3, 1, 2]
z_order = [0, 3, 1, 2, 'Idle']

p = 0.01
num_cycles = 2
error_rates = {'Meas': p, 'Prep': p, 'Idle': p, 'CNOT': p}

print('Setting up circuit level decoding matrix...')
# Construct syndrome measurement circuit
syndrome_circ = construct_sm_circuit(bb, x_order, z_order)

# print(f'Syndrome circuit is: \n{syndrome_circ}')

# Construct decoding matrix for circuit level simulation
hx_eff, short_hx_eff, hz_eff, short_hz_eff, channel_prob_x, channel_prob_z = construct_decoding_matrix(bb, syndrome_circ, error_rates, num_cycles)

# print(f'Circuit level decoding matrix constructed: \n{short_hx_eff.toarray()}')
# print(f'with channel probabilities: \nX: {channel_prob_x}')

# Generate noisy circuit
noisy_circ, err_cnt = generate_noisy_circuit(bb, syndrome_circ * num_cycles, error_rates)
print(f'Generated noisy circuit with {err_cnt} errors inserted.')

x_syndrome_history, x_state, x_syndrome_map, x_err_count = simulate_x_circuit(bb, noisy_circ + syndrome_circ + syndrome_circ)
print(f'Number of X errors: {x_err_count}')
print(f'x_syndrome_history: {x_syndrome_history}')

# Calculate true logical effect
x_state_data_qudits = [x_state[bb.qudits_dict[qudit]] for qudit in bb.data_qudits]
x_syndrome_final_logical = (np.array(lz) @ x_state_data_qudits) % field.p
print(f'True final logical X error: {x_syndrome_final_logical}')

# Syndrome sparsification
z_checks = bb.z_checks
x_syndrome_history_copy = x_syndrome_history.copy()
for check in z_checks:
    pos = x_syndrome_map[check]
    assert len(pos) == num_cycles + 2
    for row in range(1, num_cycles + 2):
        x_syndrome_history[pos[row]] += x_syndrome_history_copy[pos[row-1]]
x_syndrome_history %= field.p

# Decode
bposd = BpOsdDecoder(short_hx_eff, error_channel=channel_prob_x, max_iter=1000)
x_error = bposd.decode(x_syndrome_history)
print(f'Decoded X error: {x_error}')
assert np.all((short_hx_eff @ x_error) % field.p == x_syndrome_history)

# Check logical effect
first_logical_row = bb.l * bb.m * (num_cycles + 2)
k = len(lx)
x_syndrome_history_augmented_guessed = (hx_eff @ x_error) % field.p
x_syndrome_final_logical_guessed = x_syndrome_history_augmented_guessed[first_logical_row: first_logical_row + k]
print(f'Guessed logical effect: {x_syndrome_final_logical_guessed}')

x_success = np.array_equal(x_syndrome_final_logical_guessed, x_syndrome_final_logical)
print(f'Successful X decoding: {x_success}')
