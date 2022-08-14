# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:30:00 2022

@author: Nat Kinsky
"""
# general packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
import scipy.io as sio

# project specific packages
from session_directory import load_session_list, master_directory, make_session_list
from session_directory import find_eraser_directory as get_dir
import helpers


def load_traces(mouse: str, arena: str in ['Shock', 'Open'], day: int in [-2, -1, 0, 4, 1, 2, 7], psa: bool = False,
                list_dir: str = master_directory):
    """Load in traces and corresponding putative spiking activity if specified"""

    # Locate file directory
    make_session_list(list_dir)
    dir_use = get_dir(mouse, arena, day)

    # Import imaging data
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)
    traces = im_data['NeuronTraces'][0, 0].squeeze()[0]

    if not psa:

        return traces

    else:
        psabool = im_data['PSAbool']

        return traces, psabool


def plot_traces(traces: np.ndarray or list, psabool: np.ndarray or list = None, t: np.ndarray = None,
                SR: float or None = None, offset: float or int or None = None, ax: plt.axes or None = None,
                normalize_traces: bool = False):
    """Plot calcium traces with (optional) putative spiking activity (rising phase) in red."""

    if isinstance(traces, list):
        traces = np.array(traces)

    if ax is None:
        _, ax = plt.subplots()

    if normalize_traces:
        traces_std = np.std(traces, axis=1)
        traces = traces / traces_std[:, None]

    # Set up how much to separate each trace by in the y-direction
    if offset is None:
        offset = 10*np.mean(np.std(traces, axis=1))

    # Set up times to plot
    if t is None and SR is None:
        t = np.arange(traces.shape[1])
    elif t is None and SR is not None:
        t = np.arange(traces.shape[1])/SR

    # Plot traces
    y = 0
    for trace in traces:
        ax.plot(t, trace + y, 'k-')
        y += offset

    # Plot psa if supplied
    if psabool is not None:
        if isinstance(psabool, list):
            psabool = np.array(psabool)
        psabool = psabool.astype(bool)  # Make sure it is actually a boolean
        y = 0
        assert np.array_equal(traces.shape, psabool.shape)
        for trace, psa in zip(traces, psabool):
            idx = helpers.contiguous_regions(psa)  # Identify transients
            for id in idx:
                t_plot = t[id[0]:id[1]]
                psa_plot = trace[id[0]:id[1]]
                ax.plot(t_plot, psa_plot + y, 'r-', linewidth=plt.rcParams['lines.linewidth']*2)
            y += offset

    # Label and clean up
    ax.set_ylabel('dF/F (au)')
    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    sns.despine(ax=ax)

    pass


if __name__ == "__main__":
    traces, psabool = load_traces('Marble07', 'Shock', -1, psa=True)
    plot_traces(traces[0:10], psabool=psabool[0:10], SR=20)


