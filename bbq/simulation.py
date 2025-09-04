"""Simulation of a qudit code."""

import numpy as np
import matplotlib.pyplot as plt


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


main_theme = ["lightcoral", "lightseagreen", "royalblue"]


def plot_results(
    physical_error: dict[str, list],
    results: dict[str, list],
    num_failures: dict[str, int | list],
    noise_model: str,
    rounds: dict[str, int] = None,
    ext_physical_error: dict[str, list] = {},
    ext_results: dict[str, list] = {},
    ext_num_failures: dict[str, int | list] = {},
    colour_theme: list[str] = main_theme,
    inset_dim: list[float] = [0.54, 0.14, 0.4, 0.4],
    inset_ticks: list[float] = [0.1, 0.12, 0.14, 0.16],
    legend: list[str] = [],
    title: str = "Threshold Plot",
):
    """Plot the results of a simulation.

    Parameters
    ----------
    physical_error : dict[str, list]
        The physical error rates used in the simulation, {code : [physical_error_rates]}.
    results : dict[str, list]
        The results of the simulation, i.e. number of trials needed to hit num_failures failures for each code in plot, {code : [number_of_trials]}.
    num_failures : dict[str, int | list]
        The number of failures until simulation stopped for each code, {code : number_of_failures}.
    noise_model : str, optional
        The noise model used in the simulation, either "code_capacity" or "circuit_level".
    rounds : int, optional
        The number of rounds of stabiliser measurements for each code, i.e. {code : rounds}. Mandatory for "circuit_level" noise model.
    ext_physical_error : dict[str, list], optional
        The physical error rates used for the inset plot, {code : [physical_error_rates]}.
    ext_results : dict[str, list], optional
        The results of the simulation for the inset plot, i.e. number of trials needed to hit num_failures failures for each code in plot, {code : [number_of_trials]}.
    ext_num_failures : dict[str, int | list], optional
        The number of failures until simulation stopped for each code in the inset plot, {code : number_of_failures}.
    colour_theme : list[str], optional
        The colours to use for each code in the plot, by default main_theme.
    inset_dim : list[float], optional
        The dimensions of the inset plot, [x_location, y_location, width, height], by default [0.54, 0.14, 0.4, 0.4].
    inset_ticks : list[float], optional
        The ticks to use for the inset plot, by default [0.1, 0.12, 0.14, 0.16].
    legend : list[str], optional
        The legend labels for each code in the plot, by default the keys of the results dictionary.
    title : str, optional
        The title of the plot, by default "Threshold Plot".

    Returns
    -------
    None
    """
    plot_results, plot_error_bars = process_results(
        results, num_failures, noise_model, rounds
    )

    if not legend:
        legend = [code for code in results]
    lines = {}

    fig, ax = plt.subplots()

    physical_error = np.array(physical_error)
    ext_physical_error = np.array(ext_physical_error)

    for i, code in enumerate(results):
        res = np.array(plot_results[code])
        err = np.array(plot_error_bars[code])
        mask = np.array(results[code]) != np.inf

        (lines[code],) = ax.loglog(
            physical_error[mask], res[mask], color=colour_theme[i], label=code
        )
        ax.loglog(physical_error[mask], res[mask], ".", color=colour_theme[i])
        ax.fill_between(
            physical_error[mask],
            res[mask] - err[mask],
            res[mask] + err[mask],
            color=colour_theme[i],
            alpha=0.1,
        )

    if ext_results:
        ext_plot_results, ext_plot_error_bars = process_results(
            ext_results, ext_num_failures, noise_model, rounds
        )

        axins = ax.inset_axes(inset_dim)
        axins.set_xscale("log")

        for i, code in enumerate(ext_results):
            res = np.array(ext_plot_results[code])
            err = np.array(ext_plot_error_bars[code])
            mask = np.array(ext_results[code]) != np.inf

            axins.loglog(ext_physical_error[mask], res[mask], color=colour_theme[i])
            axins.loglog(
                ext_physical_error[mask], res[mask], ".", color=colour_theme[i]
            )
            axins.fill_between(
                ext_physical_error[mask],
                res[mask] - err[mask],
                res[mask] + err[mask],
                color=colour_theme[i],
                alpha=0.1,
            )

        axins.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            labelbottom=True,
            labelrotation=20,
        )
        axins.tick_params(
            axis="x", which="minor", bottom=False, top=False, labelbottom=False
        )
        axins.set_xticks(inset_ticks)

    ax.legend(handles=[lines[code] for code in results], labels=legend)
    ax.set_xlabel("Physical Z error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title(title)
