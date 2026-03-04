from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.decoder import BPOSD, BPMOSD, BPLSDbin, RelayBP, BP

import numpy as np
import datetime
import json
import pickle


print('Running code capacity simulation')
print('--------------------------------')

print('Setting up hx, lx...')
# Define parity check matrix, hx, and its logicals, lx, for the 3x3 qubit toric code
field = Field(5)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
# a, b = x + x**2 + y**3, y + y**2 + x**3
a, b = 1 - x, 1 - y
bb = BivariateBicycle(a, b, 5, 5, 1)
hx, lx = bb.hx, bb.x_logicals
n_qudits = hx.shape[1]
code_name = '[[72, 12, 6]]_2 RelayBP'

# Define decoder parameters for decoder (BPOSD, BPMOSD, (BPLSDbin), RelayBP)
max_iter = 100
order = 0

reset = 12
threshold = 0.001
bits_per_step = None
osd_method = 'OSD_0'

first_iter = 80
solutions = 1
relays = 10
centre = 0.125
width = 0.9

# np.random.seed(101)

# Define simulation parameters (which physical error rates to test, how many failures to observe before stopping)
physical_error = np.logspace(-0.7, -1.7, 10)
num_failures = [10 for _ in range(10)]
results = []

# Saving data
save_data = {}
save_data_filename = f'code_capacity_results_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")}.json'
save_data['code'] = bb.__repr__()
save_data['qec_code_name'] = code_name
save_data['noise_model'] = 'code_capacity'
save_data['num_failures'] = num_failures
save_data['error_rates'] = list(physical_error)
save_data['results'] = results
save_data['current_round'] = {}
json.dump(save_data, open(save_data_filename, 'w'), indent=4)

# Metrics
track_success = False
if track_success:
    save_data['only_bp'] = [0 for _ in physical_error]  # counts how many trials use BP
    save_data['bp_fail'] = [0 for _ in physical_error]  # counts how many times BP fails (when it converges)
    save_data['osd_fail'] = [0 for _ in physical_error]  # counts how many times OSD fails (when it is called)
track_priors = False
if track_priors:
    save_data['weights'] = []  # tracks error weight, when BP converges (1x success [0], 1x fails [1]) and when OSD is called (1x success [2], 1x fails [3])
    save_posteriors_filename = f'code_capacity_results_post_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")}'
    save_posteriors = []  # tracks priors, when BP converges (1x success [0], 1x fails [1]) and when OSD is called (1x success [2], 1x fails [3])

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
    # bposd = BPOSD(field, hx, x_prior, max_iter=300, order=0)
    # bplsd = BPLSDbin(field, hx, x_prior, max_iter, bits_per_step, order, osd_method)
    # bpmosd = BPMOSD(field, hx, x_prior, max_iter, order, reset, threshold)
    # bp = BP(field, hx, x_prior, max_iter=max_iter)

    # NOTE: sets mem_weight different for different powers of same qudit
    mem_weight = np.zeros((n_qudits, field.p, relays))
    mem_weight[:, :, 0] = centre
    mem_weight[:, :, 1:] = np.random.uniform(centre - width/2, centre + width/2, size=(n_qudits, field.p, relays - 1))
    # mem_weight = np.zeros((n_qudits, field.p, relays))
    # mem_weight[:, :, 0] = centre
    # rand = np.random.uniform(centre - width/2, centre + width/2, size=(n_qudits, relays - 1))
    # for i in range(0, field.p):
    #     mem_weight[:, i, 1:] = rand

    failures = 0
    num_trials = 0

    if track_priors:
        bp_s, bp_f, osd_s, osd_f = True, True, True, True
        posts = [[], [], [], []]
        weights = [[], [], [], []]

    print('Starting trials...')
    while failures < num_failures[ind]:

        # Set up decoder
        relaybp = RelayBP(field, hx, x_prior, max_iter=max_iter, first_iter=first_iter, solutions=solutions, relays=relays, mem_weight=mem_weight)
        # bp = BP(field, hx, x_prior, max_iter=1000)
        # bposd = BPOSD(field, hx, x_prior, max_iter=300, order=0)

        num_trials += 1

        # Generate syndrome
        error = np.zeros(n_qudits, dtype=int)
        error_mask = np.random.rand(n_qudits) < p
        for i in np.where(error_mask)[0]:
            error[i] = np.random.randint(1, field.p)
        syndrome = (hx @ error) % field.p

        # Decode
        if track_priors:
            if bp_s or bp_f or osd_s or osd_f:
                guessed_error, decoder_success, bp_success, posterior_track = relaybp.decode(syndrome, metric=True)
            else:
                guessed_error, decoder_success, bp_success, posterior = relaybp.decode(syndrome, metric=True)
        else:
            guessed_error, decoder_success, bp_success, posterior = relaybp.decode(syndrome, metric=False, debug=True)
        error_difference = (error - guessed_error) % field.p

        logical_effect = (np.array(lx) @ error_difference) % field.p

        if track_success:
            if bp_success:
                save_data['only_bp'][ind] += 1

        # Check success
        if np.any(logical_effect != 0):
            failures += 1
            save_data['current_round'] = {'error_rate' : p,'num_trials' : num_trials, 'failures' : failures}

            if track_success:
                if bp_success:
                    save_data['bp_fail'][ind] += 1
                else:
                    save_data['osd_fail'][ind] += 1

            if track_priors:
                if bp_success and bp_f and not (syndrome == 0).all():
                    posts[1] = posterior_track
                    weights[1] = int(sum(error_mask))
                    bp_f = False
                elif (not bp_success) and osd_f and not (syndrome == 0).all():
                    posts[3] = posterior_track
                    weights[3] = int(sum(error_mask))
                    osd_f = False

            json.dump(save_data, open(save_data_filename, 'w'), indent=4)
        else:
            if track_priors:
                if bp_success and bp_s and not (syndrome == 0).all():
                    posts[0] = posterior_track
                    weights[0] = int(sum(error_mask))
                    bp_s = False
                elif (not bp_success) and osd_s and not (syndrome == 0).all():
                    posts[2] = posterior_track
                    weights[2] = int(sum(error_mask))
                    osd_s = False

        # Update saved data
        if num_trials % 100 == 0:
            save_data['current_round'] = {'error_rate' : p,'num_trials' : num_trials, 'failures' : failures}
            json.dump(save_data, open(save_data_filename, 'w'), indent=4)

    if track_priors:
        save_posteriors.append(posts)
        save_data['weights'].append(weights)
        with open(save_posteriors_filename, 'wb') as handle:
            pickle.dump(save_posteriors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    results.append(num_trials)
    save_data['results'] = results
    json.dump(save_data, open(save_data_filename, 'w'), indent=4)

    print('Completed simulation')
    print(f'[{ind + 1}/{len(physical_error)}]')
