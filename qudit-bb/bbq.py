"""Implementation of the BivariateBicycle class for qudits."""

from polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from sympy import isprime
import warnings
import matplotlib.patches as mpatches


class ValueWarning(UserWarning):
    pass


class BivariateBicycle:
    """Implementation of the Bivariate Bicycle code on qudits.
    
    Parameters
    ----------
    a : Polynomial
        Polynomial a over the finite field.
    b : Polynomial
        Polynomial b over the finite field.
    l : int
        Dimension of left cyclic shift matrix.
    m : int
        Dimension of right cyclic shift matrix.
    q : int
        Defines CSS code construction H_x=(A|B) and H_y=(qB^T|(a.field-q)A^T).
    """

    def __init__(self, a : Polynomial, b : Polynomial, l : int, m : int, q : int):
        if not isinstance(a, Polynomial):
            raise TypeError("a must be a Polynomial")
        if not isinstance(b, Polynomial):
            raise TypeError("b must be a Polynomial")
        if not isinstance(l, int):
            raise TypeError("l must be an integer")
        if not isinstance(m, int):
            raise TypeError("m must be an integer")
        if not isinstance(q, int):
            raise TypeError("q must be an integer")
        if not 0 < q or not q < a.field:
            raise ValueError("q must be a positive integer less than the field of the polynomials")
        if a.field != b.field:
            raise ValueError("Polynomials a and b must be over the same field")
        if not isprime(a.field):
            warnings.warn("Field is not prime.", ValueWarning)
        self.a, self.b = a, b
        self.field = a.field
        self.l, self.m, self.q = l, m, q
        self.hx = np.hstack((a(l, m), b(l, m)))
        self.hz = np.hstack((q * b(l, m).transpose(), (self.field-q) * a(l, m).transpose())) % self.field
        self.A, self.B = self._monomials()
        self.qubits_dict, self.data_qubits, self.x_checks, self.z_checks = self._qubits()
        self.edges = self._edges()

    def __str__(self):
        """String representation of BivariateBicycle."""
        return f"Bivariate Bicycle code for\na(x, y) = {self.a}\nb(x, y) = {self.b}"

    def __repr__(self):
        """Canonical string epresentation of BivariateBicycle."""
        return f"BivariateBicycle({self.a.__repr__()}, {self.b.__repr__()})"

    def _monomials(self):
        """Construct monomials for the Bivariate Bicycle code."""
        a, b = self.a, self.b
        l, m = self.l, self.m
        A, B = [], []
        row, col = np.nonzero(a.coefficients)
        for i in range(len(row)):
            poly_coef = np.zeros((a.coefficients.shape), dtype=int)
            poly_coef[row[i], col[i]] = a.coefficients[row[i], col[i]]
            poly = Polynomial(a.field, poly_coef)
            A.append(poly(l, m))
        row, col = np.nonzero(b.coefficients)
        for i in range(len(row)):
            poly_coef = np.zeros((b.coefficients.shape), dtype=int)
            poly_coef[row[i], col[i]] = b.coefficients[row[i], col[i]]
            poly = Polynomial(b.field, poly_coef)
            B.append(poly(l, m))
        return A, B

    def _qubits(self):
        """Give names to each qubit and store in a dictionary: (qubit_type, qubit_type_number) : qubit_index"""
        l, m = self.l, self.m
        qubits_dict = {}
        data_qubits, x_checks, z_checks = [], [], []
        for i in range(l*m):
            # X checks
            node_name = ('x_check', i)
            x_checks.append(node_name)
            qubits_dict[node_name] = i

            # Left data qubits
            node_name = ('data_left', i)
            data_qubits.append(node_name)
            qubits_dict[node_name] = l*m + i

            # Right data qubits
            node_name = ('data_right', i)
            data_qubits.append(node_name)
            qubits_dict[node_name] = 2*l*m + i

            # Z checks
            node_name = ('z_check', i)
            z_checks.append(node_name)
            qubits_dict[node_name] = 3*l*m + i
        return qubits_dict, data_qubits, x_checks, z_checks

    def _edges(self):
        """Set up edges connecting data and measurement qubits in a dictionary: ((check_qubit_type, check_type_number), monomial_index/direction) : (qubit_type, qubit_number)"""
        l, m = self.l, self.m
        q = self.q
        field = self.field
        A, B = self.A, self.B
        edges = {}
        for i in range(l*m):
            # X checks
            check_name = ('x_check', i)
            # Left data qubits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][i, :])[0][0])
                edges[(check_name, j)] = (('data_left', y), int(A[j][i, y]))
            # Right data qubits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][i, :])[0][0])
                edges[(check_name, len(A) + j)] = (('data_right', y), int(B[j][i, y]))

            # Z checks
            check_name = ('z_check', i)
            # Left data qubits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][:, i])[0][0])
                edges[(check_name, j)] = (('data_left', y), (q * int(B[j][y, i])) % field)
            # Right data qubits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][:, i])[0][0])
                edges[(check_name, len(A) + j)] = (('data_right', y), ((field - q) * int(A[j][y, i])) % field)
        return edges

    def _simulate_z_circuit(self, circ : list):
        """Propagate a Z error through a circuit.
        
        Parameters
        ----------
        circ : list
            List of gates in circuit.
        
        Returns
        -------
        syndrome_history : nd.array
            Syndrome history, i.e. the results of the X measurements.
        state : nd.array
            Final state, 0 indicates no error, 1 indicates error.
        syndrome_map : dict
            Dictionary of {x_check qubit : list of positions in syndrome_history where qubit has been measured}.
        err_cnt : int
            Number of errors.
        """
        qubits_dict = self.qubits_dict
        field = self.field
        n = 2 * self.l * self.m

        syndrome_history, syndrome_map = [], {}
        state = np.zeros(2*n)  # Initial state with no errors
        err_cnt, syn_cnt = 0, 0
        for gate in circ:
            if gate[0] == 'CNOT':
                # IZ -> ZZ^-1, ZI -> Z^-1I
                control, target = qubits_dict[gate[1]], qubits_dict[gate[2]]
                power = gate[3]
                state[control] = (state[control] - power * state[target]) % field
                continue
            if gate[0] == 'Prep_X':
                # Reset error to 0
                qubit = qubits_dict[gate[1]]
                state[qubit] = 0
                continue
            if gate[0] == 'Meas_X':
                # Add measurement result to syndrome history
                assert gate[1][0] == 'x_check'
                qubit = qubits_dict[gate[1]]
                syndrome_history.append(state[qubit])
                if gate[1] in syndrome_map:
                    syndrome_map[gate[1]].append(syn_cnt)
                else:
                    syndrome_map[gate[1]] = [syn_cnt]
                syn_cnt += 1
                continue
            if gate[0] in ['Z', 'Y']:
                # Qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['ZX', 'YX']:
                # 1st qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['XZ', 'XY']:
                # 2nd qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[2]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['ZZ', 'YY', 'ZY', 'YZ']:
                # Both qubits gain a Z error
                err_cnt += 1
                qubit1, qubit2 = qubits_dict[gate[1]], qubits_dict[gate[2]]
                state[qubit1] = (state[qubit1] + 1) % field
                state[qubit2] = (state[qubit2] + 1) % field
                continue
        return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

    def _simulate_x_circuit(self, circ : list):
        """Propagate an X error through a circuit.
        
        Parameters
        ----------
        circ : list
            List of gates in circuit.
        
        Returns
        -------
        syndrome_history : nd.array
            Syndrome history, i.e. the results of the Z measurements.
        state : nd.array
            Final state, 0 indicates no error, 1 indicates error.
        syndrome_map : dict
            Dictionary of {z_check qubit : list of positions in syndrome_history where qubit has been measured}.
        err_cnt : int
            Number of errors.
        """
        qubits_dict = self.qubits_dict
        field = self.field
        n = 2 * self.l * self.m

        syndrome_history, syndrome_map = [], {}
        state = np.zeros(2*n)  # Initial state with no errors
        err_cnt, syn_cnt = 0, 0
        for gate in circ:
            if gate[0] == 'CNOT':
                # XI -> XX, IX -> IX
                control, target = qubits_dict[gate[1]], qubits_dict[gate[2]]
                power = gate[3]
                state[target] = (state[target] + power * state[control]) % field
                continue
            if gate[0] == 'Prep_Z':
                # Reset error to 0
                qubit = qubits_dict[gate[1]]
                state[qubit] = 0
                continue
            if gate[0] == 'Meas_Z':
                # Add measurement result to syndrome history
                assert gate[1][0] == 'z_check'
                qubit = qubits_dict[gate[1]]
                syndrome_history.append(state[qubit])
                if gate[1] in syndrome_map:
                    syndrome_map[gate[1]].append(syn_cnt)
                else:
                    syndrome_map[gate[1]] = [syn_cnt]
                syn_cnt += 1
                continue
            if gate[0] in ['X', 'Y']:
                # Qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['XZ', 'YZ']:
                # 1st qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['ZX', 'ZY']:
                # 2nd qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[2]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ['XX', 'YY', 'YX', 'XY']:
                # Both qubits gain an X error
                err_cnt += 1
                qubit1, qubit2 = qubits_dict[gate[1]], qubits_dict[gate[2]]
                state[qubit1] = (state[qubit1] + 1) % field
                state[qubit2] = (state[qubit2] + 1) % field
                continue
        return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt            

    def draw(self):
        """Draw the Bivariate Bicycle code Tanner graph."""
        # Define parameters
        hx, hz = self.hx, self.hz
        m, n = hx.shape
        a_coefficients = self.a.coefficients
        b_coefficients = self.b.coefficients
        a_factors = self.a.factor()
        b_factors = self.b.factor()

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_xlim(-0.3, (n//2)//self.l-0.2)
        ax.set_ylim(-0.3, m//self.m-0.2)
        ax.set_aspect('equal', adjustable='box')

        # Define nodes
        def x_stabiliser(x, y):
            return Rectangle((x, y), width=0.1, height=0.1, 
                        edgecolor='lightcoral', facecolor='lightcoral', zorder=3)
        def z_stabiliser(x, y):
            return Rectangle((x, y), width=0.1, height=0.1, 
                        edgecolor='lightseagreen', facecolor='lightseagreen', zorder=3)
        def l_data(x, y):
            return Circle((x, y), radius=0.06, edgecolor='royalblue', facecolor='royalblue', zorder=3)
        def r_data(x, y):
            return Circle((x, y), radius=0.06, edgecolor='gold', facecolor='gold', zorder=3)

        # Draw nodes
        for i in np.arange(0, (n//2)//self.l, 1):
            for j in np.arange(0, m//self.m, 1):
                ax.add_patch(x_stabiliser(i+0.45, j-0.05))
                ax.add_patch(z_stabiliser(i-0.05, j+0.45))
                ax.add_patch(l_data(i+0.5, j+0.5))
                ax.add_patch(r_data(i, j))

        # Draw x stabiliser edges
        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            div = a_coefficients[k, l]
                            if div == 1:
                                ax.plot([0.5+j, k+j-a_factors[0]], [i, -l+i+a_factors[1]], color='slategray')
                            else:
                                line, = ax.plot([0.5+j, k+j-a_factors[0]], [i, -l+i+a_factors[1]], color='slategray') 
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            div = b_coefficients[k, l]
                            if div == 1:
                                ax.plot([0.5+j, 0.5+k+j-b_factors[0]], [i, 0.5-l+i+b_factors[1]], color='slategray')
                            else:
                                line, = ax.plot([0.5+j, 0.5+k+j-b_factors[0]], [i, 0.5-l+i+b_factors[1]], color='slategray')  
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        # Draw z stabiliser edges
        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        if a_coefficients[k, l]:
                            div = (self.q * a_coefficients[k, l]) % self.field
                            if div == 1:
                                ax.plot([j, 0.5-k+j+a_factors[0]], [0.5+i, 0.5+l+i-a_factors[1]], color='darkgray')
                            else:
                                line, = ax.plot([j, 0.5-k+j+a_factors[0]], [0.5+i, 0.5+l+i-a_factors[1]], color='darkgray')
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        for i in range(m//self.m):
            for j in range((n//2)//self.l):
                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        if b_coefficients[k, l]:
                            div = ((self.field-self.q) * b_coefficients[k, l]) % self.field
                            if div == 1:
                                ax.plot([j, -k+j+b_factors[0]], [0.5+i, l+i-b_factors[1]], color='darkgray')
                            else:
                                line, = ax.plot([j, -k+j+b_factors[0]], [0.5+i, l+i-b_factors[1]], color='darkgray') 
                                line.set_dashes([16/div**2, 2, 16/div**2, 2])
                                line.set_dash_capstyle('round')

        # Draw boundary
        ax.plot([-0.25, -0.25], [-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, -0.25, 0, m//self.m/2, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([-0.25, (n//2)//self.l-0.25], [-0.25, -0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, -0.25, ((n//2)//self.l)/2-0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.arrow(-0.25, -0.25, ((n//2)//self.l)/2+0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([-0.25, (n//2)//self.l-0.25], [m//self.m-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow(-0.25, m//self.m-0.25, (m//self.l)/2-0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.arrow(-0.25, m//self.m-0.25, (m//self.l)/2+0.05, 0, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)
        ax.plot([(n//2)//self.l-0.25, (n//2)//self.l-0.25], [-0.25, m//self.m-0.25], color='black', linewidth=0.7)
        ax.arrow((n//2)//self.l-0.25, -0.25, 0, m//self.m/2, head_width=0.1, head_length=0.1, color='black', linewidth=0.05)

        # Make plot look nice
        ax.set_axis_off()
        ax.set_title('Tanner Graph')

        # Add legend
        handles = ['X stabiliser', 'Z stabiliser', 'Left data', 'Right data']
        lines = []
        patch_colours = ['lightcoral', 'lightseagreen', 'royalblue', 'gold']
        for i in range(4):
            lines.append(mpatches.Patch(color=patch_colours[i]))
        for i in range(1, self.field):
            xline, = ax.plot([0], [0], color='slategray')
            zline, = ax.plot([0], [0], color='darkgray')
            xline.set_dashes([16/i**2, 2, 16/i**2, 2])
            zline.set_dashes([16/i**2, 2, 16/i**2, 2])
            xline.set_dash_capstyle('round')
            zline.set_dash_capstyle('round')
            lines.append(xline)
            lines.append(zline)
            if i==1:
                handles.append('X')
                handles.append('Z')
            else:
                handles.append(f'X^{i}')
                handles.append(f'Z^{i}')
        ax.legend(lines, handles, loc='upper left', bbox_to_anchor=(1, 1), handlelength=2.4);

    def construct_sm_circuit(self, x_order : list, z_order : list) -> list:
        """Construct one cycle of the syndrome measurement circuit for the Bivariate Bicycle code.
        
        Parameters
        ----------
        x_order : list
            List of integers or 'Idle' defining the order of the CNOTs for x stabilisers.
        y_order : list
            List of integers or 'Idle' defining the order of the CNOTs for y stabilisers.
        
        Returns
        -------
        circ : list
            List of gates in one cycle of the syndrome circuit: ('CNOT', control_qubit, target_qubit, power), ('Idle', qubit), ('Meas_X', qubit), ('Meas_Z', qubit), ('Prep_X', qubit), ('Prep_Z', qubit).
        """
        if not isinstance(x_order, list):
            raise TypeError("x_order must be a list")
        if not isinstance(z_order, list):
            raise TypeError("y_order must be a list")
        for gate in x_order:
            if not (isinstance(gate, int) or gate == 'Idle'):
                raise TypeError("x_order must be an array of integers or 'Idle'")
        for gate in z_order:
            if not (isinstance(gate, int) or gate == 'Idle'):
                raise TypeError("z_order must be an array of integers or 'Idle'")
        if not x_order[0] == 'Idle':
            raise ValueError("First x_order round must be 'Idle'")
        if not z_order[-1] == 'Idle':
            raise ValueError("Last y_order round must be 'Idle'")
        for i in range(len(np.nonzero(self.hx[0])[0])):
            if i not in x_order:
                raise ValueError("x_order must contain all target qubits")
        for i in range(len(np.nonzero(self.hz[0])[0])):
            if i not in z_order:
                raise ValueError("y_order must contain all target qubits")
        if not (isinstance(num_cycles, int) and num_cycles > 0):
            raise TypeError("num_cycles must be a positive integer")
        if len(x_order) > len(z_order):
            z_order += ['Idle'] * (len(x_order) - len(z_order))
        elif len(z_order) > len(x_order):
            x_order += ['Idle'] * (len(z_order) - len(x_order))

        hx, hz = self.hx, self.hz
        a, b = self.a, self.b
        l, m, q = self.l, self.m, self.q
        field = self.field
        A, B = self.A, self.B
        qubits_dict, data_qubits, x_checks, z_checks = self.qubits_dict, self.data_qubits, self.x_checks, self.z_checks
        edges = self.edges

        # Construct the circuit
        circ = []
        U = np.identity(4*l*m, dtype=int)  # to verify CNOT order

        # For each time step, add the corresponding gate:
        # ('CNOT', control_qubit, target_qubit, power), ('Idle', qubit), ('Meas_X', qubit), ('Meas_Y', qubit), ('Prep_X', qubit)

        # Round 0: Prepare X checks, CNOT/Idle Z checks
        t = 0
        cnoted_data_qubits = []
        for qubit in x_checks:
            circ.append(('Prep_X', qubit))
        if z_order[t] == 'Idle':
            for qubit in z_checks:
                circ.append(('Idle', qubit))
        else:
            for target in z_checks:
                direction = z_order[t]
                control, power = edges[(target, direction)]
                U[qubits_dict[target], :] = (U[qubits_dict[target], :] + power * U[qubits_dict[control], :]) % field
                cnoted_data_qubits.append(control)
                circ.append(('CNOT', control, target, power))
        for qubit in data_qubits:
            if not (qubit in cnoted_data_qubits):
                circ.append(('Idle', qubit))

        # Round [1, (max-1)]: CNOT/Idle X checks, CNOT/Idle Z checks
        for t in range(1, len(x_order)-1):
            cnoted_data_qubits = []
            if x_order[t] == 'Idle':
                for qubit in x_checks:
                    circ.append(('Idle', qubit))
            else:
                for control in x_checks:
                    direction = x_order[t]
                    target, power = edges[(control, direction)]
                    U[qubits_dict[target], :] = (U[qubits_dict[target], :] + power * U[qubits_dict[control], :]) % field
                    cnoted_data_qubits.append(target)
                    circ.append(('CNOT', control, target, power))
            if z_order[t] == 'Idle':
                for qubit in z_checks:
                    circ.append(('Idle', qubit))
            else:
                for target in z_checks:
                    direction = z_order[t]
                    control, power = edges[(target, direction)]
                    U[qubits_dict[target], :] = (U[qubits_dict[target], :] + power * U[qubits_dict[control], :]) % field
                    cnoted_data_qubits.append(control)
                    circ.append(('CNOT', control, target, power))
            for qubit in data_qubits:
                if not (qubit in cnoted_data_qubits):
                    circ.append(('Idle', qubit))

        # Round max: CNOT/Idle X checks, Measure Z checks
        t = -1
        cnoted_data_qubits = []
        if x_order[t] == 'Idle':
            for qubit in x_checks:
                circ.append(('Idle', qubit))
        else:
            for control in x_checks:
                direction = x_order[t]
                target, power = edges[(control, direction)]
                U[qubits_dict[target], :] = (U[qubits_dict[target], :] + power * U[qubits_dict[control], :]) % field
                circ.append(('CNOT', control, target, power))
                cnoted_data_qubits.append(target)
        for qubit in z_checks:
            circ.append(('Meas_Z', qubit))
        for qubit in data_qubits:
            if not (qubit in cnoted_data_qubits):
                circ.append(('Idle', qubit))
        
        # Round final: Measure X checks, Prepare Z checks
        for qubit in data_qubits:
            circ.append(('Idle', qubit))
        for qubit in x_checks:
            circ.append(('Meas_X', qubit))
        for qubit in z_checks:
            circ.append(('Prep_Z', qubit))

        # Test measurement circuit against max depth circuit
        V = np.identity(4*l*m, dtype=int)
        for t in range(len(x_order)):
            if not x_order[t] == 'Idle':
                for control in x_checks:
                    direction = x_order[t]
                    target, power = edges[(control, direction)]
                    V[qubits_dict[target], :] = (V[qubits_dict[target], :] + power * V[qubits_dict[control], :]) % field
        for t in range(len(z_order)):
            if not z_order[t] == 'Idle':
                for target in z_checks:
                    direction = z_order[t]
                    control, power = edges[(target, direction)]
                    V[qubits_dict[target], :] = (V[qubits_dict[target], :] + power * V[qubits_dict[control], :]) % field
        if not np.array_equal(U, V):
            raise ValueError("Syndrome circuit does not match max depth syndrome circuit, check stabiliser orders")

        return circ
