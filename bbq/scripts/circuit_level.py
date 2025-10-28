from bbq.field import Field
from bbq.polynomial import Monomial
from bbq.bbq_code import BivariateBicycle
from bbq.circuit import construct_sm_circuit

import numpy as np

print('Running circuit level simulation')
print('--------------------------------')

print('Setting up hx, lx...')
# Define parity check matrix, hx, and its logicals, lx, for the 3x3 qubit toric code
field = Field(2)
x, y = Monomial(field, 'x'), Monomial(field, 'y')
a, b = 1 + x, 1 + y
bb = BivariateBicycle(a, b, 1, 1, 1)
hx, lx = bb.hx, bb.x_logicals
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

print(f'Syndrome circuit is: \n{syndrome_circ}')
