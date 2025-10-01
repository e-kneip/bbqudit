from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.decoder import BPOSD

import numpy as np
import datetime
import json


print('Running code capacity simulation')
print('--------------------------------')

print('Setting up hx, lx...')
# Define parity check matrix, hx, and its logicals, lx, for the 3x3 qubit toric code
field = Field(2)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 + x, 1 + y
bb = BivariateBicycle(a, b, 3, 3, 1)
hx, lx = bb.hx, bb.x_logicals
n_qudits = hx.shape[1]
code_name = '3x3 Qubit Toric Code'

# Define decoder parameters for BP+OSD
max_iter = 300
order = 0

# Define simulation parameters (which physical error rates to test, how many failures to observe before stopping)
physical_error = np.logspace(-0.7, -1.7, 5)
num_failures = 3
results = []

# Saving data
save_data = {}
save_data_filename = f'code_capacity_results_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")}.json'
save_data['qec_code_name'] = code_name
save_data['noise_model'] = 'code_capacity'
save_data['num_failures'] = num_failures
save_data['error_rates'] = list(physical_error)
save_data['results'] = results
save_data['current_round'] = {}
json.dump(save_data, open(save_data_filename, 'w'), indent=4)

print('Starting simulation...')
for ind, p in enumerate(physical_error):
    print(f'Setting up error channel for p = {p}...')
    channel_prob_x = np.ones(n_qudits) * p
    x_prior = np.zeros((n_qudits, field.p), dtype=float)

    for i, prob in enumerate(channel_prob_x):
        x_prior[i, 0] = 1 - prob
        for j in range(1, field.p):
            x_prior[i, j] = prob / (field.p - 1)

    print('Setting up decoder...')
    bposd = BPOSD(field, hx, x_prior, max_iter, order)

    failures = 0
    num_trials = 0


    print('Starting trials...')
    while failures < num_failures:
        num_trials += 1

        # Generate syndrome
        error = np.zeros(n_qudits, dtype=int)
        error_mask = np.random.rand(n_qudits) < p
        for i in np.where(error_mask)[0]:
            error[i] = np.random.randint(1, field.p)
        syndrome = (hx @ error) % field.p

        # Decode
        guessed_error, decoder_success, bp_success, posterior = bposd.decode(syndrome, debug=True)
        error_difference = (error - guessed_error) % field.p
        logical_effect = (np.array(lx) @ error_difference) % field.p

        # Check success
        if np.any(logical_effect != 0):
            failures += 1
            save_data['current_round'] = {'error_rate' : p,'num_trials' : num_trials, 'failures' : failures}
            json.dump(save_data, open(save_data_filename, 'w'), indent=4)
        elif num_trials % 100 == 0:
            save_data['current_round'] = {'error_rate' : p,'num_trials' : num_trials, 'failures' : failures}
            json.dump(save_data, open(save_data_filename, 'w'), indent=4)

    results.append(num_trials)
    save_data['results'] = results
    json.dump(save_data, open(save_data_filename, 'w'), indent=4)

    print('Completed simulation')
    print(f'[{ind + 1}/{len(physical_error)}]')
