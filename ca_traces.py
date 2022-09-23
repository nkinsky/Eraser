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
from PIL import Image
from skimage import feature

# project specific packages
from session_directory import load_session_list, master_directory, make_session_list
from session_directory import find_eraser_directory as get_dir
import placefield_stability as pfs
import helpers


def load_imaging_data(mouse: str, arena: str in ['Shock', 'Open'], day: int in [-2, -1, 0, 4, 1, 2, 7],
                      list_dir: str = master_directory):
    """Load in imaging data from FinalOutput.mat"""
    # Locate file directory
    make_session_list(list_dir)
    dir_use = get_dir(mouse, arena, day)

    # Import imaging data
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)

    return im_data


def load_ROIs(mouse, arena, day, list_dir:str = master_directory):
    """Load in ncells x nxpixels x nypixels list"""
    im_data = load_imaging_data(mouse, arena, day, list_dir=list_dir)
    return np.stack(im_data['NeuronImage'].squeeze().flatten())


def load_proj(mouse, arena, day, type: str in ['min', 'max'], list_dir: str = master_directory):
    """Load min/max projection for a session"""
    make_session_list(list_dir)
    dir_use = get_dir(mouse, arena, day)
    if type == 'min':
        im = np.array(Image.open(path.join(dir_use, 'ICMovie_min_proj.tif')))
    elif type == 'max':
        im = np.array(Image.open(path.join(dir_use, 'ICMovie_max_proj.tif')))

    return im


def plot_ROIs(rois, bkgrd: np.ndarray or bool = True, color: str = 'r', ax=None):
    """Plot all rois in desired color. bkgrd can be a max/min projection, or True = plot over white background
    the size of the imaging FOV. False = just plot with no background."""

   # Create white background if not provided
    if not isinstance(bkgrd, np.ndarray) and bkgrd:
        bkgrd = np.ones_like(rois[1])  # Make background white

    # Create axes if not specified
    if ax is None:
        _, ax = plt.subplots()

    # Plot bkgrd if specified
    if isinstance(bkgrd, np.ndarray):
        ax.imshow(bkgrd, cmap='gray')

    # Detect edges and plot neurons
    for roi in rois:
        xedges, yedges = detect_roi_edges(roi)
        ax.plot(xedges, yedges, color=color)

    # Remove everything
    ax.axis('off')

    return ax


def plot_ROIs_bw_sessions(mouse, arena1, day1, arena2, day2, proj: str = 'min', ax=None):
    """Plot ROIs from two sessions in different colors with co-active cells in green.

    Currently only works for same-day sessions - does not register ROIs between sessions due to
    affine transformation data being loading from MATLAB into python improperly"""

    # Load ROIs and projection from first session
    rois1 = load_ROIs(mouse, arena1, day1)
    if proj in ['min', 'max']:
        bkgrd = load_proj(mouse, arena1, day1, proj)

    # Load rois from second session
    rois2 = load_ROIs(mouse, arena2, day2)

    # Load map between two sessions
    neuron_map = pfs.get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=True)

    if ax is None:
        _, ax = plt.subplots()

    # Now plot ROIs
    plot_ROIs(rois1, bkgrd=bkgrd, color='g', ax=ax)  # First session
    plot_ROIs(rois2, bkgrd=False, color='y', ax=ax)  # Second session
    plot_ROIs(rois1[neuron_map >= 0], bkgrd=False, color='r', ax=ax)  # Overlapping ROIs

    return ax


def detect_roi_edges(roi_binary):
    """Detect roi edges and organize them nicely in CW/CCW fashion"""
    edges = feature.canny(roi_binary)  # detect edges
    inds = np.where(edges) # Get edge locations in pixels
    isort = np.argsort(np.arctan2(inds[1] - inds[1].mean(), inds[0] - inds[0].mean()))  # Sort by angle from center

    xedges = np.append(inds[1][isort], inds[1][isort[0]])
    yedges = np.append(inds[0][isort], inds[0][isort[0]])

    return xedges, yedges


def load_traces(mouse: str, arena: str in ['Shock', 'Open'], day: int in [-2, -1, 0, 4, 1, 2, 7], psa: bool = False,
                list_dir: str = master_directory):
    """Load in traces and corresponding putative spiking activity if specified"""

    im_data = load_imaging_data(mouse, arena, day, list_dir=list_dir)
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


