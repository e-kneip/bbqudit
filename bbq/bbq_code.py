"""Implementation of the BivariateBicycle class for qudits."""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.patches as mpatches
import galois

from scipy.sparse import coo_matrix, hstack
from matplotlib.patches import Rectangle, Circle
from sympy import isprime
from bbq.polynomial import Polynomial


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
    name : str
        Name of the code.
    """

    def __init__(
        self, a: Polynomial, b: Polynomial, l: int, m: int, q: int, name: str = None
    ):
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
            raise ValueError(
                "q must be a positive integer less than the field of the polynomials"
            )
        if a.field != b.field:
            raise ValueError("Polynomials a and b must be over the same field")
        if not isprime(a.field):
            print("Warning: Field is not prime.")
            warnings.warn("Field is not prime.", ValueWarning)
        if not (isinstance(name, str) or name == None):
            raise TypeError("name must be a string")
        self.a, self.b = a, b
        self.field = a.field
        self.l, self.m, self.q = l, m, q
        self.hx = np.hstack((a(l, m), b(l, m)))
        self.hz = (
            np.hstack((q * b(l, m).transpose(), (self.field - q) * a(l, m).transpose()))
            % self.field
        )
        self.A, self.B = self._monomials()
        self.qubits_dict, self.data_qubits, self.x_checks, self.z_checks = (
            self._qubits()
        )
        self.edges = self._edges()
        self.name = name
        self.x_logicals, self.z_logicals = self._compute_logicals()
        if not self.x_logicals:
            print("Warning: No X logicals found for these parameters.")
            warnings.warn("No X logicals found for these parameters.", ValueWarning)
        if not self.z_logicals:
            print("Warning: No Z logicals found for these parameters.")
            warnings.warn("No Z logicals found for these parameters.", ValueWarning)

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
        for i in range(l * m):
            # X checks
            node_name = ("x_check", i)
            x_checks.append(node_name)
            qubits_dict[node_name] = i

            # Left data qubits
            node_name = ("data_left", i)
            data_qubits.append(node_name)
            qubits_dict[node_name] = l * m + i

            # Right data qubits
            node_name = ("data_right", i)
            data_qubits.append(node_name)
            qubits_dict[node_name] = 2 * l * m + i

            # Z checks
            node_name = ("z_check", i)
            z_checks.append(node_name)
            qubits_dict[node_name] = 3 * l * m + i
        return qubits_dict, data_qubits, x_checks, z_checks

    def _edges(self):
        """Set up edges connecting data and measurement qubits in a dictionary: ((check_qubit_type, check_type_number), monomial_index/direction) : (qubit_type, qubit_number)"""
        l, m = self.l, self.m
        q = self.q
        field = self.field
        A, B = self.A, self.B
        edges = {}
        for i in range(l * m):
            # X checks
            check_name = ("x_check", i)
            # Left data qubits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][i, :])[0][0])
                edges[(check_name, j)] = (("data_left", y), int(A[j][i, y]))
            # Right data qubits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][i, :])[0][0])
                edges[(check_name, len(A) + j)] = (("data_right", y), int(B[j][i, y]))

            # Z checks
            check_name = ("z_check", i)
            # Left data qubits
            for j in range(len(B)):
                y = int(np.nonzero(B[j][:, i])[0][0])
                edges[(check_name, j)] = (
                    ("data_left", y),
                    (q * int(B[j][y, i])) % field,
                )
            # Right data qubits
            for j in range(len(A)):
                y = int(np.nonzero(A[j][:, i])[0][0])
                edges[(check_name, len(A) + j)] = (
                    ("data_right", y),
                    ((field - q) * int(A[j][y, i])) % field,
                )
        return edges

    def _compute_logicals(self):
        """Compute logical operators for the code."""
        hx, hz = self.hx, self.hz
        field = self.field

        # Set up Galois field array
        GF = galois.GF(field)
        Hx_gal, Hz_gal = GF(hx), GF(hz)
        x_logicals, z_logicals = [], []
        x_check, z_check = Hx_gal, Hz_gal

        # X logicals must be in the kernel of Hz and not the image of Hx^T
        ker_hz = Hz_gal.null_space()
        rank = np.linalg.matrix_rank(Hx_gal)
        for vec in ker_hz:
            x_check = GF(np.vstack((x_check, vec)))
            if np.linalg.matrix_rank(x_check) > rank:
                x_logicals.append(vec)
                rank += 1
            else:
                np.delete(x_check, -1, axis=0)

        # Z logicals must be in the kernel of Hx and not the image of Hz^T
        ker_hx = Hx_gal.null_space()
        rank = np.linalg.matrix_rank(Hz_gal)
        for vec in ker_hx:
            z_check = GF(np.vstack((z_check, vec)))
            if np.linalg.matrix_rank(z_check) > rank:
                z_logicals.append(vec)
                rank += 1
            else:
                np.delete(z_check, -1, axis=0)

        # Check correct number of logicals found: k = n - m
        assert len(x_logicals) == len(z_logicals)
        m = np.linalg.matrix_rank(Hx_gal) + np.linalg.matrix_rank(Hz_gal)
        n = self.hx.shape[1]
        if not len(x_logicals) == n - m:
            raise ValueError("Incorrect number of logical operators found.")

        return [x_log.__array__(dtype=int) for x_log in x_logicals], [
            z_log.__array__(dtype=int) for z_log in z_logicals
        ]

    def _simulate_z_circuit(self, circ: list):
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
        state = np.zeros(2 * n, dtype=int)  # Initial state with no errors
        err_cnt, syn_cnt = 0, 0
        for gate in circ:
            if gate[0] == "CNOT":
                # IZ -> ZZ^-1, ZI -> Z^-1I
                control, target = qubits_dict[gate[1]], qubits_dict[gate[2]]
                power = gate[3]
                state[control] = (state[control] - power * state[target]) % field
                continue
            if gate[0] == "Prep_X":
                # Reset error to 0
                qubit = qubits_dict[gate[1]]
                state[qubit] = 0
                continue
            if gate[0] == "Meas_X":
                # Add measurement result to syndrome history
                assert gate[1][0] == "x_check"
                qubit = qubits_dict[gate[1]]
                syndrome_history.append(state[qubit])
                if gate[1] in syndrome_map:
                    syndrome_map[gate[1]].append(syn_cnt)
                else:
                    syndrome_map[gate[1]] = [syn_cnt]
                syn_cnt += 1
                continue
            if gate[0] in ["Z", "Y"]:
                # Qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["ZX", "YX"]:
                # 1st qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["XZ", "XY"]:
                # 2nd qubit gains a Z error
                err_cnt += 1
                qubit = qubits_dict[gate[2]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["ZZ", "YY", "ZY", "YZ"]:
                # Both qubits gain a Z error
                err_cnt += 1
                qubit1, qubit2 = qubits_dict[gate[1]], qubits_dict[gate[2]]
                state[qubit1] = (state[qubit1] + 1) % field
                state[qubit2] = (state[qubit2] + 1) % field
                continue
        return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

    def _simulate_x_circuit(self, circ: list):
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
        state = np.zeros(2 * n, dtype=int)  # Initial state with no errors
        err_cnt, syn_cnt = 0, 0
        for gate in circ:
            if gate[0] == "CNOT":
                # XI -> XX, IX -> IX
                control, target = qubits_dict[gate[1]], qubits_dict[gate[2]]
                power = gate[3]
                state[target] = (state[target] + power * state[control]) % field
                continue
            if gate[0] == "Prep_Z":
                # Reset error to 0
                qubit = qubits_dict[gate[1]]
                state[qubit] = 0
                continue
            if gate[0] == "Meas_Z":
                # Add measurement result to syndrome history
                assert gate[1][0] == "z_check"
                qubit = qubits_dict[gate[1]]
                syndrome_history.append(state[qubit])
                if gate[1] in syndrome_map:
                    syndrome_map[gate[1]].append(syn_cnt)
                else:
                    syndrome_map[gate[1]] = [syn_cnt]
                syn_cnt += 1
                continue
            if gate[0] in ["X", "Y"]:
                # Qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["XZ", "YZ"]:
                # 1st qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[1]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["ZX", "ZY"]:
                # 2nd qubit gains an X error
                err_cnt += 1
                qubit = qubits_dict[gate[2]]
                state[qubit] = (state[qubit] + 1) % field
                continue
            if gate[0] in ["XX", "YY", "YX", "XY"]:
                # Both qubits gain an X error
                err_cnt += 1
                qubit1, qubit2 = qubits_dict[gate[1]], qubits_dict[gate[2]]
                state[qubit1] = (state[qubit1] + 1) % field
                state[qubit2] = (state[qubit2] + 1) % field
                continue
        return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

    def _generate_noisy_circuit(self, circ, error_rates):
        """Generate circuit with noise, i.e. insert errors wrt error_rates dict.

        Parameters
        ----------
        circ : list
            List of gates in the circuit.
        error_rates : dict
            Dictionary with error rates with keys ['Meas', 'Prep', 'Idle', 'CNOT'].

        Returns
        -------
        noisy_circ : list
            List of gates in the circuit with errors.
        err_cnt : int
            Number of errors inserted.
        """
        noisy_circ = []
        err_cnt = 0
        field = self.field
        for gate in circ:
            assert gate[0] in [
                "CNOT",
                "Prep_X",
                "Prep_Z",
                "Meas_X",
                "Meas_Z",
                "Idle",
            ], "Invalid gate type."
            if gate[0] == "Meas_X":
                # Meas_X error only affects Z stabilisers
                if np.random.uniform() <= error_rates["Meas"]:
                    # Random Z^k error for k = 1, 2, ..., field-1
                    power = np.random.randint(field - 1)
                    noisy_circ += [("Z", gate[1])] * (power + 1)
                    err_cnt += 1
                noisy_circ.append(gate)
                continue
            if gate[0] == "Meas_Z":
                # Meas_Z error only affects X stabilisers
                if np.random.uniform() <= error_rates["Meas"]:
                    # Random X^k error for k = 1, 2, ..., field-1
                    power = np.random.randint(field - 1)
                    noisy_circ += [("X", gate[1])] * (power + 1)
                    err_cnt += 1
                noisy_circ.append(gate)
                continue
            if gate[0] == "Prep_X":
                # Prep_X error only affects Z stabilisers
                noisy_circ.append(gate)
                if np.random.uniform() <= error_rates["Prep"]:
                    # Random Z^k error for k = 1, 2, ..., field-1
                    power = np.random.randint(field - 1)
                    noisy_circ += [("Z", gate[1])] * (power + 1)
                    err_cnt += 1
                continue
            if gate[0] == "Prep_Z":
                # Prep_Z error only affects X stabilisers
                noisy_circ.append(gate)
                if np.random.uniform() <= error_rates["Prep"]:
                    # Random X^k error for k = 1, 2, ..., field-1
                    power = np.random.randint(field - 1)
                    noisy_circ += [("X", gate[1])] * (power + 1)
                    err_cnt += 1
                continue
            if gate[0] == "Idle":
                # Idle error can be X^k, Y^k or Z^k
                if np.random.uniform() <= error_rates["Idle"]:
                    ptype = np.random.randint(3)
                    if ptype == 0:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("X", gate[1])] * (power + 1)
                    elif ptype == 1:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Y", gate[1])] * (power + 1)
                    else:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Z", gate[1])] * (power + 1)
                    err_cnt += 1
                continue
            if gate[0] == "CNOT":
                # CNOT error can be X^k, Y^k, Z^k or combinations of them
                noisy_circ.append(gate)
                if np.random.uniform() <= error_rates["CNOT"]:
                    err_cnt += 1
                    ptype = np.random.randint(15)
                    if ptype == 0:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("X", gate[1])] * (power + 1)
                        continue
                    if ptype == 1:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Y", gate[1])] * (power + 1)
                        continue
                    if ptype == 2:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Z", gate[1])] * (power + 1)
                        continue
                    if ptype == 3:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("X", gate[2])] * (power + 1)
                        continue
                    if ptype == 4:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Y", gate[2])] * (power + 1)
                        continue
                    if ptype == 5:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("Z", gate[2])] * (power + 1)
                        continue
                    if ptype == 6:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("XX", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 7:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("YY", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 8:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("ZZ", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 9:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("XY", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 10:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("YX", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 11:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("YZ", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 12:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("ZY", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 13:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("XZ", gate[1], gate[2])] * (power + 1)
                        continue
                    if ptype == 14:
                        power = np.random.randint(field - 1)
                        noisy_circ += [("ZX", gate[1], gate[2])] * (power + 1)
                        continue
        return noisy_circ, err_cnt

    def draw(self):
        """Draw the Bivariate Bicycle code Tanner graph."""
        # Define parameters
        hx, hz = self.hx, self.hz
        m, n = hx.shape
        a_coefficients, b_coefficients = self.a.coefficients, self.b.coefficients
        a_factors_min, a_factors_max = self.a.factor()
        b_factors_min, b_factors_max = self.b.factor()
        x_max = max(a_factors_max[0], b_factors_max[0])
        y_max = max(a_factors_max[1], b_factors_max[1])
        name = self.name

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_xlim(-0.3, (n // 2) // self.l - 0.2)
        ax.set_ylim(-0.3, m // self.m - 0.2)
        ax.set_aspect("equal", adjustable="box")

        # Define nodes
        def x_stabiliser(x, y):
            return Rectangle(
                (x, y),
                width=0.1,
                height=0.1,
                edgecolor="lightcoral",
                facecolor="lightcoral",
                zorder=3,
            )

        def z_stabiliser(x, y):
            return Rectangle(
                (x, y),
                width=0.1,
                height=0.1,
                edgecolor="lightseagreen",
                facecolor="lightseagreen",
                zorder=3,
            )

        def l_data(x, y):
            return Circle(
                (x, y),
                radius=0.06,
                edgecolor="royalblue",
                facecolor="royalblue",
                zorder=3,
            )

        def r_data(x, y):
            return Circle(
                (x, y), radius=0.06, edgecolor="gold", facecolor="gold", zorder=3
            )

        # Draw nodes
        for i in np.arange(0, (n // 2) // self.l, 1):
            for j in np.arange(0, m // self.m, 1):
                ax.add_patch(x_stabiliser(i + 0.45, j - 0.05))
                ax.add_patch(z_stabiliser(i - 0.05, j + 0.45))
                ax.add_patch(l_data(i + 0.5, j + 0.5))
                ax.add_patch(r_data(i, j))

        for i in range(-x_max, x_max + m // self.m):
            for j in range(-y_max, y_max + (n // 2) // self.l):
                for k in range(a_coefficients.shape[0]):
                    for l in range(a_coefficients.shape[1]):
                        # Draw x stabiliser edges
                        if a_coefficients[k, l]:
                            div = a_coefficients[k, l]
                            if div == 1:
                                ax.plot(
                                    [0.5 + j, k + j - a_factors_min[0]],
                                    [i, -l + i + a_factors_min[1]],
                                    color="slategray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [0.5 + j, k + j - a_factors_min[0]],
                                    [i, -l + i + a_factors_min[1]],
                                    color="slategray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                        # Draw z stabiliser edges
                        if a_coefficients[k, l]:
                            div = (self.q * a_coefficients[k, l]) % self.field
                            if div == 1:
                                ax.plot(
                                    [j, 0.5 - k + j + a_factors_min[0]],
                                    [0.5 + i, 0.5 + l + i - a_factors_min[1]],
                                    color="darkgray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [j, 0.5 - k + j + a_factors_min[0]],
                                    [0.5 + i, 0.5 + l + i - a_factors_min[1]],
                                    color="darkgray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                for k in range(b_coefficients.shape[0]):
                    for l in range(b_coefficients.shape[1]):
                        # Draw x stabiliser edges
                        if b_coefficients[k, l]:
                            div = b_coefficients[k, l]
                            if div == 1:
                                ax.plot(
                                    [0.5 + j, 0.5 + k + j - b_factors_min[0]],
                                    [i, 0.5 - l + i + b_factors_min[1]],
                                    color="slategray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [0.5 + j, 0.5 + k + j - b_factors_min[0]],
                                    [i, 0.5 - l + i + b_factors_min[1]],
                                    color="slategray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

                        # Draw z stabiliser edges
                        if b_coefficients[k, l]:
                            div = (
                                (self.field - self.q) * b_coefficients[k, l]
                            ) % self.field
                            if div == 1:
                                ax.plot(
                                    [j, -k + j + b_factors_min[0]],
                                    [0.5 + i, l + i - b_factors_min[1]],
                                    color="darkgray",
                                )
                            else:
                                (line,) = ax.plot(
                                    [j, -k + j + b_factors_min[0]],
                                    [0.5 + i, l + i - b_factors_min[1]],
                                    color="darkgray",
                                )
                                line.set_dashes([16 / div**2, 2, 16 / div**2, 2])
                                line.set_dash_capstyle("round")

        # Draw boundary
        ax.plot(
            [-0.25, -0.25], [-0.25, m // self.m - 0.25], color="black", linewidth=0.7
        )
        ax.arrow(
            -0.25,
            -0.25,
            0,
            m // self.m / 2,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [-0.25, (n // 2) // self.l - 0.25],
            [-0.25, -0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            -0.25,
            -0.25,
            ((n // 2) // self.l) / 2 - 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.arrow(
            -0.25,
            -0.25,
            ((n // 2) // self.l) / 2 + 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [-0.25, (n // 2) // self.l - 0.25],
            [m // self.m - 0.25, m // self.m - 0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            -0.25,
            m // self.m - 0.25,
            (m // self.l) / 2 - 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.arrow(
            -0.25,
            m // self.m - 0.25,
            (m // self.l) / 2 + 0.05,
            0,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )
        ax.plot(
            [(n // 2) // self.l - 0.25, (n // 2) // self.l - 0.25],
            [-0.25, m // self.m - 0.25],
            color="black",
            linewidth=0.7,
        )
        ax.arrow(
            (n // 2) // self.l - 0.25,
            -0.25,
            0,
            m // self.m / 2,
            head_width=0.1,
            head_length=0.1,
            color="black",
            linewidth=0.05,
        )

        # Make plot look nice
        ax.set_axis_off()
        if name:
            ax.set_title(f"Tanner Graph of {name}")
        else:
            ax.set_title("Tanner Graph")

        # Add legend
        handles = ["X stabiliser", "Z stabiliser", "Left data", "Right data"]
        lines = []
        patch_colours = ["lightcoral", "lightseagreen", "royalblue", "gold"]
        for i in range(4):
            lines.append(mpatches.Patch(color=patch_colours[i]))
        for i in range(1, self.field):
            (xline,) = ax.plot([0], [0], color="slategray")
            (zline,) = ax.plot([0], [0], color="darkgray")
            xline.set_dashes([16 / i**2, 2, 16 / i**2, 2])
            zline.set_dashes([16 / i**2, 2, 16 / i**2, 2])
            xline.set_dash_capstyle("round")
            zline.set_dash_capstyle("round")
            lines.append(xline)
            lines.append(zline)
            if i == 1:
                handles.append("X")
                handles.append("Z")
            else:
                handles.append(f"X^{i}")
                handles.append(f"Z^{i}")
        ax.legend(
            lines, handles, loc="upper left", bbox_to_anchor=(1, 1), handlelength=2.4
        )

    def construct_sm_circuit(self, x_order: list, z_order: list) -> list:
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
            if not (isinstance(gate, int) or gate == "Idle"):
                raise TypeError("x_order must be an array of integers or 'Idle'")
        for gate in z_order:
            if not (isinstance(gate, int) or gate == "Idle"):
                raise TypeError("z_order must be an array of integers or 'Idle'")
        if not x_order[0] == "Idle":
            raise ValueError("First x_order round must be 'Idle'")
        if not z_order[-1] == "Idle":
            raise ValueError("Last y_order round must be 'Idle'")
        for i in range(len(np.nonzero(self.hx[0])[0])):
            if i not in x_order:
                raise ValueError("x_order must contain all target qubits")
        for i in range(len(np.nonzero(self.hz[0])[0])):
            if i not in z_order:
                raise ValueError("y_order must contain all target qubits")
        if len(x_order) > len(z_order):
            z_order += ["Idle"] * (len(x_order) - len(z_order))
        elif len(z_order) > len(x_order):
            x_order += ["Idle"] * (len(z_order) - len(x_order))

        hx, hz = self.hx, self.hz
        a, b = self.a, self.b
        l, m, q = self.l, self.m, self.q
        field = self.field
        A, B = self.A, self.B
        qubits_dict, data_qubits, x_checks, z_checks = (
            self.qubits_dict,
            self.data_qubits,
            self.x_checks,
            self.z_checks,
        )
        edges = self.edges

        # Construct the circuit
        circ = []
        U = np.identity(4 * l * m, dtype=int)  # to verify CNOT order

        # For each time step, add the corresponding gate:
        # ('CNOT', control_qubit, target_qubit, power), ('Idle', qubit), ('Meas_X', qubit), ('Meas_Y', qubit), ('Prep_X', qubit)

        # Round 0: Prepare X checks, CNOT/Idle Z checks
        t = 0
        cnoted_data_qubits = []
        for qubit in x_checks:
            circ.append(("Prep_X", qubit))
        if z_order[t] == "Idle":
            for qubit in z_checks:
                circ.append(("Idle", qubit))
        else:
            for target in z_checks:
                direction = z_order[t]
                control, power = edges[(target, direction)]
                U[qubits_dict[target], :] = (
                    U[qubits_dict[target], :] + power * U[qubits_dict[control], :]
                ) % field
                cnoted_data_qubits.append(control)
                circ.append(("CNOT", control, target, power))
        for qubit in data_qubits:
            if not (qubit in cnoted_data_qubits):
                circ.append(("Idle", qubit))

        # Round [1, (max-1)]: CNOT/Idle X checks, CNOT/Idle Z checks
        for t in range(1, len(x_order) - 1):
            cnoted_data_qubits = []
            if x_order[t] == "Idle":
                for qubit in x_checks:
                    circ.append(("Idle", qubit))
            else:
                for control in x_checks:
                    direction = x_order[t]
                    target, power = edges[(control, direction)]
                    U[qubits_dict[target], :] = (
                        U[qubits_dict[target], :] + power * U[qubits_dict[control], :]
                    ) % field
                    cnoted_data_qubits.append(target)
                    circ.append(("CNOT", control, target, power))
            if z_order[t] == "Idle":
                for qubit in z_checks:
                    circ.append(("Idle", qubit))
            else:
                for target in z_checks:
                    direction = z_order[t]
                    control, power = edges[(target, direction)]
                    U[qubits_dict[target], :] = (
                        U[qubits_dict[target], :] + power * U[qubits_dict[control], :]
                    ) % field
                    cnoted_data_qubits.append(control)
                    circ.append(("CNOT", control, target, power))
            for qubit in data_qubits:
                if not (qubit in cnoted_data_qubits):
                    circ.append(("Idle", qubit))

        # Round max: CNOT/Idle X checks, Measure Z checks
        t = -1
        cnoted_data_qubits = []
        if x_order[t] == "Idle":
            for qubit in x_checks:
                circ.append(("Idle", qubit))
        else:
            for control in x_checks:
                direction = x_order[t]
                target, power = edges[(control, direction)]
                U[qubits_dict[target], :] = (
                    U[qubits_dict[target], :] + power * U[qubits_dict[control], :]
                ) % field
                circ.append(("CNOT", control, target, power))
                cnoted_data_qubits.append(target)
        for qubit in z_checks:
            circ.append(("Meas_Z", qubit))
        for qubit in data_qubits:
            if not (qubit in cnoted_data_qubits):
                circ.append(("Idle", qubit))

        # Round final: Measure X checks, Prepare Z checks
        for qubit in data_qubits:
            circ.append(("Idle", qubit))
        for qubit in x_checks:
            circ.append(("Meas_X", qubit))
        for qubit in z_checks:
            circ.append(("Prep_Z", qubit))

        # Test measurement circuit against max depth circuit
        V = np.identity(4 * l * m, dtype=int)
        for t in range(len(x_order)):
            if not x_order[t] == "Idle":
                for control in x_checks:
                    direction = x_order[t]
                    target, power = edges[(control, direction)]
                    V[qubits_dict[target], :] = (
                        V[qubits_dict[target], :] + power * V[qubits_dict[control], :]
                    ) % field
        for t in range(len(z_order)):
            if not z_order[t] == "Idle":
                for target in z_checks:
                    direction = z_order[t]
                    control, power = edges[(target, direction)]
                    V[qubits_dict[target], :] = (
                        V[qubits_dict[target], :] + power * V[qubits_dict[control], :]
                    ) % field
        if not np.array_equal(U, V):
            raise ValueError(
                "Syndrome circuit does not match max depth syndrome circuit, check stabiliser orders"
            )

        return circ

    def construct_decoding_matrix(
        self, circ: list, error_rates: dict, num_cycles: int = 1
    ) -> np.ndarray:
        """Construct decoding matrix for a given syndrome circuit.

        Parameters
        ----------
        circ : list
            List of gates in one cycle of the syndrome circuit: ('CNOT', control_qubit, target_qubit, power), ('Idle', qubit), ('Meas_X', qubit), ('Meas_Z', qubit), ('Prep_X', qubit), ('Prep_Z', qubit).
        error_rate : dict
            Dictionary of error rates for keys [Meas, Prep, Idle, CNOT].
        num_cycles : int
            Number of cycles to repeat the syndrome circuit. Default is 1.

        Returns
        -------
        hx_eff : coo_matrix
            Decoding matrix for X stabilisers.
        short_hx_eff : coo_matrix
            Decoding matrix for X stabilisers without columns for logicals.
        hz_eff : coo_matrix
            Decoding matrix for Z stabilisers.
        short_hz_eff : coo_matrix
            Decoding matrix for Z stabilisers without columns for logicals.
        channel_prob_x : list
            List of probabilities for each X syndrome, i.e. each column in hx_eff.
        channel_prob_z : list
            List of probabilities for each Z syndrome, i.e. each column in hz_eff.
        """
        if not (isinstance(error_rates, dict)):
            raise TypeError("error_rates must be a dictionary")
        for key in error_rates.keys():
            if (key not in ["Meas", "Prep", "Idle", "CNOT"]) or (len(error_rates) != 4):
                raise ValueError(
                    "error_rates must have keys ['Meas', 'Prep', 'Idle', 'CNOT']"
                )
            if not (isinstance(error_rates[key], float) and 0 <= error_rates[key] <= 1):
                raise ValueError("error_rates must have values between 0 and 1")
        if not (isinstance(num_cycles, int) and num_cycles > 0):
            raise TypeError("num_cycles must be a positive integer")

        l, m = self.l, self.m
        field = self.field
        qubits_dict, data_qubits = self.qubits_dict, self.data_qubits
        x_logicals, z_logicals = self.x_logicals, self.z_logicals
        x_checks, z_checks = self.x_checks, self.z_checks

        # Construct repeated circuit
        repeated_circ = circ * num_cycles

        # Single error circuits
        z_prob, z_circuit = [], []
        x_prob, x_circuit = [], []
        head = []
        tail = repeated_circ.copy()
        for gate in repeated_circ:
            # assert gate[0] in ['CNOT', 'Idle', 'Meas_X', 'Meas_Z', 'Prep_X', 'Prep_Z']
            if gate[0] == "Meas_X":
                # Meas_X error only affects Z detectors
                z_circuit.append(head + [("Z", gate[1])] + tail)
                z_prob.append(error_rates["Meas"])
            if gate[0] == "Meas_Z":
                # Meas_Z error only affects X detectors
                x_circuit.append(head + [("X", gate[1])] + tail)
                x_prob.append(error_rates["Meas"])
            head.append(gate)
            tail.pop(0)
            # assert repeated_circ == head + tail
            if gate[0] == "Prep_X":
                # Prep_X error only affects Z detectors
                z_circuit.append(head + [("Z", gate[1])] + tail)
                z_prob.append(error_rates["Prep"])
            if gate[0] == "Prep_Z":
                # Prep_Z error only affects X detectors
                x_circuit.append(head + [("X", gate[1])] + tail)
                x_prob.append(error_rates["Prep"])
            if gate[0] == "Idle":
                # Idle error on Z detectors
                z_circuit.append(head + [("Z", gate[1])] + tail)
                z_prob.append(
                    error_rates["Idle"] * 2 / 3
                )  # 3 possible Idle errors are X, Y, Z so Z is 2/3 (Y and Z)
                # Idle error on X detectors
                x_circuit.append(head + [("X", gate[1])] + tail)
                x_prob.append(error_rates["Idle"] * 2 / 3)
            if gate[0] == "CNOT":
                # Z error on control
                z_circuit.append(head + [("Z", gate[1])] + tail)
                z_prob.append(
                    error_rates["CNOT"] * 4 / 15
                )  # possible CNOT errors are IX, IY, ..., ZZ so Z is 4/15 (IZ, IY, XZ and XY)
                # Z error on target
                z_circuit.append(head + [("Z", gate[2])] + tail)
                z_prob.append(error_rates["CNOT"] * 4 / 15)
                # Z error on both
                z_circuit.append(head + [("ZZ", gate[1], gate[2])] + tail)
                z_prob.append(error_rates["CNOT"] * 4 / 15)
                # X error on control
                x_circuit.append(head + [("X", gate[1])] + tail)
                x_prob.append(error_rates["CNOT"] * 4 / 15)
                # X error on target
                x_circuit.append(head + [("X", gate[2])] + tail)
                x_prob.append(error_rates["CNOT"] * 4 / 15)
                # X error on both
                x_circuit.append(head + [("XX", gate[1], gate[2])] + tail)
                x_prob.append(error_rates["CNOT"] * 4 / 15)

        # Execute each noisy X circuit and compute syndrome
        # Add two noiseless syndrome cycles to end
        cnt = 0
        Hx_dict = {}
        for x_circ in x_circuit:
            syndrome_history, state, syndrome_map, err_cnt = self._simulate_x_circuit(
                x_circ + circ + circ
            )
            assert err_cnt == 1
            assert len(syndrome_history) == l * m * (num_cycles + 2)

            # Compute final state of data qubits and logical effect
            state_data_qubits = [
                state[qubits_dict[qubit]] for qubit in data_qubits
            ]  # 1 indicates X error
            syndrome_final_logical = (
                np.array(z_logicals) @ state_data_qubits
            ) % field  # Check if X error flips logical Z outcome

            # Syndrome sparsification, i.e. only keep syndrome entries that change from previous cycle
            syndrome_history_copy = syndrome_history.copy()
            for check in z_checks:
                pos = syndrome_map[check]
                assert len(pos) == num_cycles + 2
                for row in range(1, num_cycles + 2):
                    syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
            syndrome_history %= field

            # Combine syndrome_history and syndrome_final_logical
            syndrome_history_augmented = np.hstack(
                [syndrome_history, syndrome_final_logical]
            )

            # Hx_dict maps flagged Z stabilisers to corresponding noisy circuit, i.e. Hx_dict[flagged_z_stab] = [noisy_circuit_1, noisy_circuit_2, ...]
            supp = tuple(np.nonzero(syndrome_history_augmented)[0])
            if supp in Hx_dict:
                Hx_dict[supp].append(cnt)
            else:
                Hx_dict[supp] = [cnt]
            cnt += 1

        first_logical_row_x = l * m * (num_cycles + 2)
        num_x_errors = len(Hx_dict)  # Number of distinct X syndrome histories
        k = len(x_logicals)  # Number of logical qubits
        hx_eff, short_hx_eff = [], []
        channel_prob_x = []
        for supp in Hx_dict:
            new_col = np.zeros(
                (l * m * (num_cycles + 2) + k, 1), dtype=int
            )  # With the augmented part for logicals
            new_col_short = np.zeros((l * m * (num_cycles + 2), 1), dtype=int)
            new_col[list(supp), 0] = 1  # 1 indicates which stabiliser is flagged
            new_col_short[:, 0] = new_col[0:first_logical_row_x, 0]
            hx_eff.append(coo_matrix(new_col))
            short_hx_eff.append(coo_matrix(new_col_short))
            channel_prob_x.append(
                np.sum([x_prob[i] for i in Hx_dict[supp]])
            )  # Probability of a given X syndrome
        hx_eff = hstack(
            hx_eff
        )  # Row = flagged detectors (+ logical effect), column = eror mechanism (with same logical effect)
        short_hx_eff = hstack(
            short_hx_eff
        )  # Shortened hx_eff without rows for logicals

        # Execute each noisy Z circuit and compute syndrome
        # Add two noiseless syndrome cycles to end
        cnt = 0
        Hz_dict = {}
        for z_circ in z_circuit:
            syndrome_history, state, syndrome_map, err_cnt = self._simulate_z_circuit(
                z_circ + circ + circ
            )
            assert err_cnt == 1
            assert len(syndrome_history) == l * m * (num_cycles + 2)

            # Compute final state of data qubits and logical effect
            state_data_qubits = [state[qubits_dict[qubit]] for qubit in data_qubits]
            syndrome_final_logical = (np.array(x_logicals) @ state_data_qubits) % field

            # Syndrome sparsification, i.e. only keep syndrome entries that change from previous cycle
            syndrome_history_copy = syndrome_history.copy()
            for check in x_checks:
                pos = syndrome_map[check]
                assert len(pos) == num_cycles + 2
                for row in range(1, num_cycles + 2):
                    syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
            syndrome_history %= field

            # Combine syndrome_history and syndrome_final_logical
            syndrome_history_augmented = np.hstack(
                [syndrome_history, syndrome_final_logical]
            )

            # Hz_dict maps flagged X stabilisers to corresponding noisy circuit, i.e. Hz_dict[flagged_x_stab] = [noisy_circuit_1, noisy_circuit_2, ...]
            supp = tuple(np.nonzero(syndrome_history_augmented)[0])
            if supp in Hz_dict:
                Hz_dict[supp].append(cnt)
            else:
                Hz_dict[supp] = [cnt]
            cnt += 1

        first_logical_row_z = l * m * (num_cycles + 2)
        num_z_errors = len(Hz_dict)  # Number of distinct Z syndrome histories
        hz_eff, short_hz_eff = [], []
        channel_prob_z = []
        for supp in Hz_dict:
            new_col = np.zeros(
                (l * m * (num_cycles + 2) + k, 1), dtype=int
            )  # With the augmented part for logicals
            new_col_short = np.zeros((l * m * (num_cycles + 2), 1), dtype=int)
            new_col[list(supp), 0] = 1  # 1 indicates which stabiliser is flagged
            new_col_short[:, 0] = new_col[0:first_logical_row_z, 0]
            hz_eff.append(coo_matrix(new_col))
            short_hz_eff.append(coo_matrix(new_col_short))
            channel_prob_z.append(
                np.sum([z_prob[i] for i in Hz_dict[supp]])
            )  # Probability of a given Z syndrome
        hz_eff = hstack(
            hz_eff
        )  # Row = flagged detectors (+ logical effect), column = eror mechanism (with same logical effect)
        short_hz_eff = hstack(
            short_hz_eff
        )  # Shortened hz_eff without rows for logicals

        return (
            hx_eff,
            short_hx_eff,
            hz_eff,
            short_hz_eff,
            channel_prob_x,
            channel_prob_z,
        )
