"""Circuit level noise model construction for Bivariate Bicycle codes."""

from bbq.bbq_code import BivariateBicycle

import numpy as np
from scipy.sparse import coo_matrix, hstack

def construct_sm_circuit(code: BivariateBicycle, x_order: list[int | str], z_order: list[int | str]) -> list:
    """Construct one cycle of the syndrome measurement circuit for the Bivariate Bicycle code.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    x_order : list
        List of integers or 'Idle' defining the order of the CNOTs for x stabilisers.
    y_order : list
        List of integers or 'Idle' defining the order of the CNOTs for y stabilisers.

    Returns
    -------
    circ : list
        List of gates in one cycle of the syndrome circuit: ('CNOT', control_qudit, target_qudit, power), ('Idle', qudit), ('Meas_X', qudit), ('Meas_Z', qudit), ('Prep_X', qudit), ('Prep_Z', qudit).
    """
    for gate in x_order:
        if not (isinstance(gate, int) or gate == "Idle"):
            raise TypeError("x_order must be an array of integers or 'Idle'")
    for gate in z_order:
        if not (isinstance(gate, int) or gate == "Idle"):
            raise TypeError("z_order must be an array of integers or 'Idle'")
    if not x_order[0] == "Idle":
        raise ValueError("First x_order round must be 'Idle'")
    if not z_order[-1] == "Idle":
        raise ValueError("Last z_order round must be 'Idle'")

    for i in range(len(np.nonzero(code.hx[0])[0])):
        if i not in x_order:
            raise ValueError("x_order must contain all target qudits")
    for i in range(len(np.nonzero(code.hz[0])[0])):
        if i not in z_order:
            raise ValueError("z_order must contain all target qudits")

    if len(x_order) > len(z_order):
        z_order += ["Idle"] * (len(x_order) - len(z_order))
    elif len(z_order) > len(x_order):
        x_order += ["Idle"] * (len(z_order) - len(x_order))

    l, m = code.l, code.m
    field = code.field
    qudits_dict, data_qudits, x_checks, z_checks = (
        code.qudits_dict,
        code.data_qudits,
        code.x_checks,
        code.z_checks,
    )
    edges = code.edges

    # Construct the circuit
    circ = []
    U = np.identity(4 * l * m, dtype=int)  # to verify CNOT order

    # For each time step, add the corresponding gate:
    # ('CNOT', control_qudit, target_qudit, power), ('Idle', qudit), ('Meas_X', qudit), ('Meas_Z', qudit), ('Prep_X', qudit), ('Prep_Z', qudit)

    # Round 0: Prepare X checks, CNOT/Idle Z checks
    t = 0
    cnoted_data_qudits = []
    for qudit in x_checks:
        circ.append(("Prep_X", qudit))
    if z_order[t] == "Idle":
        for qudit in z_checks:
            circ.append(("Idle", qudit))
    else:
        for target in z_checks:
            direction = z_order[t]
            control, power = edges[(target, direction)]
            U[qudits_dict[target], :] = (
                U[qudits_dict[target], :] + power * U[qudits_dict[control], :]
            ) % field.p
            cnoted_data_qudits.append(control)
            circ.append(("CNOT", control, target, power))
    for qudit in data_qudits:
        if not (qudit in cnoted_data_qudits):
            circ.append(("Idle", qudit))

    # Round [1, (max-1)]: CNOT/Idle X checks, CNOT/Idle Z checks
    for t in range(1, len(x_order) - 1):
        cnoted_data_qudits = []
        if x_order[t] == "Idle":
            for qudit in x_checks:
                circ.append(("Idle", qudit))
        else:
            for control in x_checks:
                direction = x_order[t]
                target, power = edges[(control, direction)]
                U[qudits_dict[target], :] = (
                    U[qudits_dict[target], :] + power * U[qudits_dict[control], :]
                ) % field.p
                cnoted_data_qudits.append(target)
                circ.append(("CNOT", control, target, power))
        if z_order[t] == "Idle":
            for qudit in z_checks:
                circ.append(("Idle", qudit))
        else:
            for target in z_checks:
                direction = z_order[t]
                control, power = edges[(target, direction)]
                U[qudits_dict[target], :] = (
                    U[qudits_dict[target], :] + power * U[qudits_dict[control], :]
                ) % field.p
                cnoted_data_qudits.append(control)
                circ.append(("CNOT", control, target, power))
        for qudit in data_qudits:
            if not (qudit in cnoted_data_qudits):
                circ.append(("Idle", qudit))

    # Round max: CNOT/Idle X checks, Measure Z checks
    t = -1
    cnoted_data_qudits = []
    if x_order[t] == "Idle":
        for qudit in x_checks:
            circ.append(("Idle", qudit))
    else:
        for control in x_checks:
            direction = x_order[t]
            target, power = edges[(control, direction)]
            U[qudits_dict[target], :] = (
                U[qudits_dict[target], :] + power * U[qudits_dict[control], :]
            ) % field.p
            circ.append(("CNOT", control, target, power))
            cnoted_data_qudits.append(target)
    for qudit in z_checks:
        circ.append(("Meas_Z", qudit))
    for qudit in data_qudits:
        if not (qudit in cnoted_data_qudits):
            circ.append(("Idle", qudit))

    # Round final: Measure X checks, Prepare Z checks
    for qudit in data_qudits:
        circ.append(("Idle", qudit))
    for qudit in x_checks:
        circ.append(("Meas_X", qudit))
    for qudit in z_checks:
        circ.append(("Prep_Z", qudit))

    # Test measurement circuit against max depth circuit
    V = np.identity(4 * l * m, dtype=int)
    for t in range(len(x_order)):
        if not x_order[t] == "Idle":
            for control in x_checks:
                direction = x_order[t]
                target, power = edges[(control, direction)]
                V[qudits_dict[target], :] = (
                    V[qudits_dict[target], :] + power * V[qudits_dict[control], :]
                ) % field.p
    for t in range(len(z_order)):
        if not z_order[t] == "Idle":
            for target in z_checks:
                direction = z_order[t]
                control, power = edges[(target, direction)]
                V[qudits_dict[target], :] = (
                    V[qudits_dict[target], :] + power * V[qudits_dict[control], :]
                ) % field.p
    if not np.array_equal(U, V):
        raise ValueError(
            "Syndrome circuit does not match max depth syndrome circuit, check stabiliser orders"
        )

    return circ

def generate_single_error_circuits(repeated_circ: list[tuple], error_rates: dict[str, float]) -> tuple[list[float], list[tuple], list[float], list[tuple]]:
    """
    Generate all single error circuits for a syndrome measurement circuit.
    
    Parameters
    ----------
    repeated_circ : list[tuple]
        List of gates in the repeated syndrome circuit.
    error_rates : dict[str, float]
        Dictionary of error rates for keys [Meas, Prep, Idle, CNOT].
    
    Returns
    -------
    x_prob : list[float]
        List of probabilities for each X error circuit.
    x_circuit : list[tuple]
        List of X error circuits.
    z_prob : list[float]
        List of probabilities for each Z error circuit.
    z_circuit : list[tuple]
        List of Z error circuits.
    """

    # Set up single error circuits
    z_prob, z_circuit = [], []
    x_prob, x_circuit = [], []

    head = []
    tail = repeated_circ.copy()

    for gate in repeated_circ:
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
    return x_prob, x_circuit, z_prob, z_circuit

def simulate_z_circuit(code: BivariateBicycle, circ: list):
    """Propagate a Z error through a circuit.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    circ : list
        List of gates in circuit.

    Returns
    -------
    syndrome_history : nd.array
        Syndrome history, i.e. the results of the X measurements.
    state : nd.array
        Final state, 0 indicates no error, 1 indicates error.
    syndrome_map : dict
        Dictionary of {x_check qudit : list of positions in syndrome_history where qudit has been measured}.
    err_cnt : int
        Number of errors.
    """
    qudits_dict = code.qudits_dict
    field = code.field
    n = 2 * code.l * code.m

    syndrome_history, syndrome_map = [], {}
    state = np.zeros(2 * n, dtype=int)  # Initial state with no errors
    err_cnt, syn_cnt = 0, 0
    for gate in circ:
        if gate[0] == "CNOT":
            # IZ -> ZZ^-1, ZI -> Z^-1I
            control, target = qudits_dict[gate[1]], qudits_dict[gate[2]]
            power = gate[3]
            state[control] = (state[control] - power * state[target]) % field.p
            continue
        if gate[0] == "Prep_X":
            # Reset error to 0
            qudit = qudits_dict[gate[1]]
            state[qudit] = 0
            continue
        if gate[0] == "Meas_X":
            # Add measurement result to syndrome history
            assert gate[1][0] == "x_check"
            qudit = qudits_dict[gate[1]]
            syndrome_history.append(state[qudit])
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt += 1
            continue
        if gate[0] in ["Z", "Y"]:
            # qudit gains a Z error
            err_cnt += 1
            qudit = qudits_dict[gate[1]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["ZX", "YX"]:
            # 1st qudit gains a Z error
            err_cnt += 1
            qudit = qudits_dict[gate[1]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["XZ", "XY"]:
            # 2nd qudit gains a Z error
            err_cnt += 1
            qudit = qudits_dict[gate[2]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["ZZ", "YY", "ZY", "YZ"]:
            # Both qudits gain a Z error
            err_cnt += 1
            qudit1, qudit2 = qudits_dict[gate[1]], qudits_dict[gate[2]]
            state[qudit1] = (state[qudit1] + 1) % field.p
            state[qudit2] = (state[qudit2] + 1) % field.p
            continue
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

def simulate_x_circuit(code: BivariateBicycle, circ: list):
    """Propagate an X error through a circuit.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    circ : list
        List of gates in circuit.

    Returns
    -------
    syndrome_history : nd.array
        Syndrome history, i.e. the results of the Z measurements.
    state : nd.array
        Final state, 0 indicates no error, 1 indicates error.
    syndrome_map : dict
        Dictionary of {z_check qudit : list of positions in syndrome_history where qudit has been measured}.
    err_cnt : int
        Number of errors.
    """
    qudits_dict = code.qudits_dict
    field = code.field
    n = 2 * code.l * code.m

    syndrome_history, syndrome_map = [], {}
    state = np.zeros(2 * n, dtype=int)  # Initial state with no errors
    err_cnt, syn_cnt = 0, 0
    for gate in circ:
        if gate[0] == "CNOT":
            # XI -> XX, IX -> IX
            control, target = qudits_dict[gate[1]], qudits_dict[gate[2]]
            power = gate[3]
            state[target] = (state[target] + power * state[control]) % field.p
            continue
        if gate[0] == "Prep_Z":
            # Reset error to 0
            qudit = qudits_dict[gate[1]]
            state[qudit] = 0
            continue
        if gate[0] == "Meas_Z":
            # Add measurement result to syndrome history
            assert gate[1][0] == "z_check"
            qudit = qudits_dict[gate[1]]
            syndrome_history.append(state[qudit])
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt += 1
            continue
        if gate[0] in ["X", "Y"]:
            # qudit gains an X error
            err_cnt += 1
            qudit = qudits_dict[gate[1]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["XZ", "YZ"]:
            # 1st qudit gains an X error
            err_cnt += 1
            qudit = qudits_dict[gate[1]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["ZX", "ZY"]:
            # 2nd qudit gains an X error
            err_cnt += 1
            qudit = qudits_dict[gate[2]]
            state[qudit] = (state[qudit] + 1) % field.p
            continue
        if gate[0] in ["XX", "YY", "YX", "XY"]:
            # Both qudits gain an X error
            err_cnt += 1
            qudit1, qudit2 = qudits_dict[gate[1]], qudits_dict[gate[2]]
            state[qudit1] = (state[qudit1] + 1) % field.p
            state[qudit2] = (state[qudit2] + 1) % field.p
            continue
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

def build_hx_dict(code: BivariateBicycle, x_circuit: list[tuple], circ: list[tuple], num_cycles: int) -> dict[tuple[int], list[int]]:
    """
    Build hx_dict mapping syndromes to noisy circuits.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    x_circuit : list[tuple]
        List of noisy X error circuits.
    circ : list[tuple]
        List of gates in one cycle of the syndrome circuit.
    num_cycles : int
        Number of cycles to repeat the syndrome circuit.
    
    Returns
    -------
    hx_dict : dict
        Dictionary mapping flagged Z stabilisers to corresponding noisy circuits.
    """
    # Execute each noisy X circuit and compute syndrome
    # Add two noiseless syndrome cycles to end
    cnt = 0
    hx_dict = {}
    for x_circ in x_circuit:
        syndrome_history, state, syndrome_map, err_cnt = simulate_x_circuit(
            code, x_circ + circ + circ
        )
        assert err_cnt == 1
        assert len(syndrome_history) == code.l * code.m * (num_cycles + 2)

        # Compute final state of data qudits and logical effect
        state_data_qudits = [
            state[code.qudits_dict[qudit]] for qudit in code.data_qudits
        ]  # 1 indicates X error
        syndrome_final_logical = (
            np.array(code.z_logicals) @ state_data_qudits
        ) % code.field.p  # Check if X error flips logical Z outcome

        # Syndrome sparsification, i.e. only keep syndrome entries that change from previous cycle
        syndrome_history_copy = syndrome_history.copy()
        for check in code.z_checks:
            pos = syndrome_map[check]
            assert len(pos) == num_cycles + 2
            for row in range(1, num_cycles + 2):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= code.field.p

        # Combine syndrome_history and syndrome_final_logical
        syndrome_history_augmented = np.hstack(
            [syndrome_history, syndrome_final_logical]
        )

        # Hx_dict maps flagged Z stabilisers to corresponding noisy circuit, i.e. Hx_dict[flagged_z_stab] = [noisy_circuit_1, noisy_circuit_2, ...]
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in hx_dict:
            hx_dict[supp].append(cnt)
        else:
            hx_dict[supp] = [cnt]
        cnt += 1
    return hx_dict

def build_hz_dict(code: BivariateBicycle, z_circuit: list[tuple], circ: list[tuple], num_cycles: int) -> dict[tuple[int], list[int]]:
    """
    Build hz_dict mapping syndromes to noisy circuits.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    z_circuit : list[tuple]
        List of noisy Z error circuits.
    circ : list[tuple]
        List of gates in one cycle of the syndrome circuit.
    num_cycles : int
        Number of cycles to repeat the syndrome circuit.

    Returns
    -------
    hz_dict : dict
        Dictionary mapping flagged Z stabilisers to corresponding noisy circuits.
    """
    # Execute each noisy Z circuit and compute syndrome
    # Add two noiseless syndrome cycles to end
    cnt = 0
    Hz_dict = {}
    for z_circ in z_circuit:
        syndrome_history, state, syndrome_map, err_cnt = simulate_z_circuit(
            code, z_circ + circ + circ
        )
        assert err_cnt == 1
        assert len(syndrome_history) == code.l * code.m * (num_cycles + 2)

        # Compute final state of data qudits and logical effect
        state_data_qudits = [state[code.qudits_dict[qudit]] for qudit in code.data_qudits]
        syndrome_final_logical = (
            np.array(code.x_logicals) @ state_data_qudits
        ) % code.field.p

        # Syndrome sparsification, i.e. only keep syndrome entries that change from previous cycle
        syndrome_history_copy = syndrome_history.copy()
        for check in code.x_checks:
            pos = syndrome_map[check]
            assert len(pos) == num_cycles + 2
            for row in range(1, num_cycles + 2):
                syndrome_history[pos[row]] += syndrome_history_copy[pos[row - 1]]
        syndrome_history %= code.field.p

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
    return Hz_dict

def build_hx_eff(code: BivariateBicycle, hx_dict: dict[tuple[int], list[int]], x_prob: list[float], num_cycles: int) -> tuple[coo_matrix[int], coo_matrix[int], list[float]]:
    """
    Build hx_eff decoding matrix from hx_dict.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    hx_dict : dict
        Dictionary mapping flagged Z stabilisers to corresponding noisy circuits.
    x_prob : list[float]
        List of probabilities for each X error circuit.
    num_cycles : int
        Number of cycles to repeat the syndrome circuit.

    Returns
    -------
    hx_eff : coo_matrix[int]
        Decoding matrix for X stabilisers with logical effect.
    short_hx_eff : coo_matrix[int]
        Decoding matrix for X stabilisers.
    channel_prob_x : list[float]
        List of probabilities for each X syndrome, i.e. each column in hx_eff.
    """
    first_logical_row_x = code.l * code.m * (num_cycles + 2)
    k = len(code.x_logicals)  # Number of logical qudits
    hx_eff, short_hx_eff = [], []
    channel_prob_x = []
    for supp in hx_dict:
        new_col = np.zeros(
            (code.l * code.m * (num_cycles + 2) + k, 1), dtype=int
        )  # With the augmented part for logicals
        new_col_short = np.zeros((code.l * code.m * (num_cycles + 2), 1), dtype=int)
        new_col[list(supp), 0] = 1  # 1 indicates which stabiliser is flagged
        new_col_short[:, 0] = new_col[0:first_logical_row_x, 0]
        hx_eff.append(coo_matrix(new_col))
        short_hx_eff.append(coo_matrix(new_col_short))
        channel_prob_x.append(
            np.sum([x_prob[i] for i in hx_dict[supp]])
        )  # Probability of a given X syndrome
    hx_eff = hstack(
        hx_eff
    )  # Row = flagged detectors (+ logical effect), column = eror mechanism (with same logical effect)
    short_hx_eff = hstack(
        short_hx_eff
    )  # Shortened hx_eff without rows for logicals

    return hx_eff, short_hx_eff, channel_prob_x

def build_hz_eff(code: BivariateBicycle, Hz_dict: dict[tuple[int], list[int]], z_prob: list[float], num_cycles: int) -> tuple[coo_matrix[int], coo_matrix[int], list[float]]:
    """
    Build hz_eff decoding matrix from Hz_dict.
    
    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    Hz_dict : dict[tuple[int], list[int]]
        Dictionary mapping flagged X stabilisers to corresponding noisy circuits.
    z_prob : list[float]
        List of probabilities for each Z error circuit.
    num_cycles : int
        Number of cycles to repeat the syndrome circuit.
    
    Returns
    -------
    hz_eff : coo_matrix[int]
        Decoding matrix for Z stabilisers with logical effect.
    short_hz_eff : coo_matrix[int]
        Decoding matrix for Z stabilisers.
    channel_prob_z : list[float]
        List of probabilities for each Z syndrome, i.e. each column in hz_eff.
    """
    first_logical_row_z = code.l * code.m * (num_cycles + 2)
    k = len(code.z_logicals)  # Number of logical qudits
    hz_eff, short_hz_eff = [], []
    channel_prob_z = []
    for supp in Hz_dict:
        new_col = np.zeros(
            (code.l * code.m * (num_cycles + 2) + k, 1), dtype=int
        )  # With the augmented part for logicals
        new_col_short = np.zeros((code.l * code.m * (num_cycles + 2), 1), dtype=int)
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
    return hz_eff, short_hz_eff, channel_prob_z

def construct_decoding_matrix(
    code: BivariateBicycle, circ: list[tuple], error_rates: dict[str, float], num_cycles: int = 1
) -> tuple[coo_matrix[int], coo_matrix[int], coo_matrix[int], coo_matrix[int], list[float], list[float]]:
    """Construct decoding matrix for a given syndrome circuit.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    circ : list[tuple]
        List of gates in one cycle of the syndrome circuit: ('CNOT', control_qudit, target_qudit, power), ('Idle', qudit), ('Meas_X', qudit), ('Meas_Z', qudit), ('Prep_X', qudit), ('Prep_Z', qudit).
    error_rate : dict[str, float]
        Dictionary of error rates for keys [Meas, Prep, Idle, CNOT].
        Dictionary of error rates for keys [Meas, Prep, Idle, CNOT].
    num_cycles : int
        Number of cycles to repeat the syndrome circuit. Default is 1.

    Returns
    -------
    hx_eff : coo_matrix[int]
        Decoding matrix for X stabilisers.
    short_hx_eff : coo_matrix[int]
        Decoding matrix for X stabilisers without columns for logicals.
    hz_eff : coo_matrix[int]
        Decoding matrix for Z stabilisers.
    short_hz_eff : coo_matrix[int]
        Decoding matrix for Z stabilisers without columns for logicals.
    channel_prob_x : list[float]
        List of probabilities for each X syndrome, i.e. each column in hx_eff.
    channel_prob_z : list[float]
        List of probabilities for each Z syndrome, i.e. each column in hz_eff.
    """
    for key in error_rates.keys():
        if (key not in ["Meas", "Prep", "Idle", "CNOT"]) or (len(error_rates) != 4):
            raise ValueError(
                "error_rates must have keys ['Meas', 'Prep', 'Idle', 'CNOT']"
            )
        if not 0 <= error_rates[key] <= 1:
            raise ValueError("error_rates must have values between 0 and 1")
    if not num_cycles > 0:
        raise TypeError("num_cycles must be a positive integer")

    # Construct all possible single error circuits of the repeated syndrome circuit
    repeated_circ = circ * num_cycles
    x_prob, x_circuit, z_prob, z_circuit = generate_single_error_circuits(repeated_circ, error_rates)

    # Execute each noisy X/Z circuit and compute syndromes
    hx_dict = build_hx_dict(code, x_circuit, circ, num_cycles)
    hz_dict = build_hz_dict(code, z_circuit, circ, num_cycles)

    # Build hx_eff, hz_eff decoding matrix
    hx_eff, short_hx_eff, channel_prob_x = build_hx_eff(code, hx_dict, x_prob, num_cycles)
    hz_eff, short_hz_eff, channel_prob_z = build_hz_eff(code, hz_dict, z_prob, num_cycles)

    return (
        hx_eff,
        short_hx_eff,
        hz_eff,
        short_hz_eff,
        channel_prob_x,
        channel_prob_z,
    )

def generate_noisy_circuit(code: BivariateBicycle, circ: list[tuple], error_rates: dict[str, float]) -> tuple[list[tuple], int]:
    """Generate circuit with noise, i.e. insert errors wrt error_rates dict.

    Parameters
    ----------
    code : BivariateBicycle
        The Bivariate Bicycle code to simulate.
    circ : list[tuple]
        List of gates in the circuit.
    error_rates : dict[str, float]
        Dictionary with error rates with keys ['Meas', 'Prep', 'Idle', 'CNOT'].

    Returns
    -------
    noisy_circ : list[tuple]
        List of gates in the circuit with errors.
    err_cnt : int
        Number of errors inserted.
    """
    noisy_circ = []
    err_cnt = 0
    field = code.field
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
                # Random Z^k error for k = 1, 2, ..., field.p-1
                power = np.random.randint(field.p - 1)
                noisy_circ += [("Z", gate[1])] * (power + 1)
                err_cnt += 1
            noisy_circ.append(gate)
            continue
        if gate[0] == "Meas_Z":
            # Meas_Z error only affects X stabilisers
            if np.random.uniform() <= error_rates["Meas"]:
                # Random X^k error for k = 1, 2, ..., field.p-1
                power = np.random.randint(field.p - 1)
                noisy_circ += [("X", gate[1])] * (power + 1)
                err_cnt += 1
            noisy_circ.append(gate)
            continue
        if gate[0] == "Prep_X":
            # Prep_X error only affects Z stabilisers
            noisy_circ.append(gate)
            if np.random.uniform() <= error_rates["Prep"]:
                # Random Z^k error for k = 1, 2, ..., field.p-1
                power = np.random.randint(field.p - 1)
                noisy_circ += [("Z", gate[1])] * (power + 1)
                err_cnt += 1
            continue
        if gate[0] == "Prep_Z":
            # Prep_Z error only affects X stabilisers
            noisy_circ.append(gate)
            if np.random.uniform() <= error_rates["Prep"]:
                # Random X^k error for k = 1, 2, ..., field.p-1
                power = np.random.randint(field.p - 1)
                noisy_circ += [("X", gate[1])] * (power + 1)
                err_cnt += 1
            continue
        if gate[0] == "Idle":
            # Idle error can be X^k, Y^k or Z^k
            if np.random.uniform() <= error_rates["Idle"]:
                ptype = np.random.randint(3)
                if ptype == 0:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("X", gate[1])] * (power + 1)
                elif ptype == 1:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("Y", gate[1])] * (power + 1)
                else:
                    power = np.random.randint(field.p - 1)
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
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("X", gate[1])] * (power + 1)
                    continue
                if ptype == 1:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("Y", gate[1])] * (power + 1)
                    continue
                if ptype == 2:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("Z", gate[1])] * (power + 1)
                    continue
                if ptype == 3:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("X", gate[2])] * (power + 1)
                    continue
                if ptype == 4:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("Y", gate[2])] * (power + 1)
                    continue
                if ptype == 5:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("Z", gate[2])] * (power + 1)
                    continue
                if ptype == 6:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("XX", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 7:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("YY", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 8:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("ZZ", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 9:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("XY", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 10:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("YX", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 11:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("YZ", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 12:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("ZY", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 13:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("XZ", gate[1], gate[2])] * (power + 1)
                    continue
                if ptype == 14:
                    power = np.random.randint(field.p - 1)
                    noisy_circ += [("ZX", gate[1], gate[2])] * (power + 1)
                    continue
    return noisy_circ, err_cnt
