from bbq.simulation import process_results
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
filenames = ["code_capacity_results_17-09-2025_15-16.json"]
legend = []
noise_model = "code_capacity"
rounds = None
num_failures = {}
results = {}
ext_results = {}
title = "3x3 Qubit Toric Code"

colour_theme = ["lightcoral", "lightseagreen", "royalblue"]
inset_dim: list[float] = [0.54, 0.14, 0.4, 0.4]
inset_ticks: list[float] = [0.1, 0.12, 0.14, 0.16]

for filename in filenames:
    save_data = json.load(open(filename, 'r'))
    code = save_data['qec_code_name']
    legend.append(code)
    num_failures[code] = save_data['num_failures']
    physical_error = save_data['error_rates']  # assume same physical error rates for all files
    results[code] = save_data['results']

# Process results
plot_results, plot_error_bars = process_results(results, num_failures, noise_model)

lines = {}

fig, ax = plt.subplots()

physical_error = np.array(physical_error)
if ext_results:
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
plt.show()
