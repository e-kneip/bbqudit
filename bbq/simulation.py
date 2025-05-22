"""Simulation of a qudit code."""

import numpy as np


def process_results(
    results: dict, num_failures: dict, noise_model: str, rounds: dict = None
) -> tuple[dict, dict]:
    """Process the results of a simulation using code capacity.

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
