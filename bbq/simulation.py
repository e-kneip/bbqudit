"""Simulation of a qudit code."""

import numpy as np


def process_results(
    results: dict,
    num_failures: dict,
    noise_model: str,
    rounds: dict = None,
) -> tuple[dict, dict]:
    """Process the results of a simulation.

    Parameters
    ----------
    results : dict
        The results of the simulation, i.e. {code : [number_of_trials]}.
    num_failures : dict
        The number of failures for each code, i.e. {code : number_of_failures}.
    noise_model : str, optional
        The noise model used in the simulation, either "code_capacity" or "circuit_level".
    rounds : dict, optional
        The number of rounds of stabiliser measurements for each code, i.e. {code : rounds}. Mandatory for "circuit_level" noise model.

    Returns
    -------
    plot_results : dict
        The logical error rate per round from the data, i.e. {code : [logical_error_rate_per_round]}.
    plot_error_bars : dict
        The error bars from the data, i.e. {code : [error_bar]}.
    """

    if noise_model == "circuit_level":
        if rounds is None:
            raise ValueError("rounds must be provided for circuit_level noise model")
        for d in results:
            results[d] = np.array(results[d]) * rounds[d]

    plot_results = {}
    plot_error_bars = {}

    for d in results:
        plot_results[d] = num_failures[d] / np.array(results[d])
        plot_error_bars[d] = np.sqrt(
            (plot_results[d]) * (1 - plot_results[d]) / results[d]
        )

    return plot_results, plot_error_bars


def generate_syndrome():
    pass


def simulate(
    field: int,
    h: np.ndarray,
    logicals: np.ndarray,
    noise_model: str,
    num_failures: int = 1,
    rounds: int = None,
    filename: str = None,
) -> dict:
    """Simulate the qudit code under the given noise model.

    Parameters
    ----------
    field : int
        The dimension of the qudit code.
    h : np.ndarray
        The parity check matrix of the code.
    logicals : np.ndarray
        The logical operators of the code.
    noise_model : str
        The noise model to use, either "code_capacity" or "circuit_level".
    num_failures : int, optional
        The number of failures to stop simulation at, by default 1.
    rounds : int, optional
        The number of rounds of stabiliser measurements, by default None. Mandatory for "circuit_level" noise model.
    filename : str, optional
        The filename to save the results to, by default None will set name to results_date.

    Returns
    -------
    save_data : dict
        The results of the simulation, i.e. {'current_round' : dict, 'noise_model' : str, 'rounds' : int, 'num_failures' : int, 'error_rates' : np.ndarray, 'results' : list}.
    """
    if not isinstance(field, int) or field <= 0:
        raise ValueError("field must be a positive integer")
    if not isinstance(h, np.ndarray):
        raise TypeError("h must be a numpy array")
    if not isinstance(logicals, np.ndarray):
        raise TypeError("logicals must be a numpy array")
    if noise_model not in ["code_capacity", "circuit_level"]:
        raise ValueError(
            "noise_model must be either 'code_capacity' or 'circuit_level'"
        )
    if noise_model == "circuit_level" and rounds is None:
        raise ValueError("rounds must be provided for circuit_level noise model")
    if not isinstance(num_failures, int) or num_failures <= 0:
        raise ValueError("num_failures must be a positive integer")
    if not (isinstance(rounds, int) and rounds > 0) and rounds is not None:
        raise TypeError("rounds must be a positive integer or None")
    if filename is not None and not isinstance(filename, str):
        raise TypeError("filename must be a string")

    # Generate syndrome
    # Decode
    # Check for logical error
    # Repeat until num_failures is reached

    # May want different functions for different noise models
    # Simulate just x errors or do both?

    pass


def plot_results():
    pass
