# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:30:00 2022

@author: Nat Kinsky
"""
# general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
import scipy.io as sio
from PIL import Image
from skimage import feature, measure
from pathlib import Path
import scipy.ndimage as ndi

# project specific packages
from session_directory import load_session_list, master_directory, make_session_list, find_eraser_session
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


def get_roi_orientations(rois):
    """Get roi orientations"""

    orientations = [measure.regionprops((roi > 0).astype(int))[0].orientation for roi in rois]

    return np.array(orientations)


def calc_orientation_diff_bw_sessions(mouse, arena1, day1, arena2, day2, batch_map_use=True):
    """Calculate change in neuron ROI orientation between sessions as a metric of cell registration quality"""

    # Load ROIs
    rois1 = load_ROIs(mouse, arena1, day1)
    rois2 = load_ROIs(mouse, arena2, day2)

    # Get ROI orientation
    orient1 = get_roi_orientations(rois1)
    orient2 = get_roi_orientations(rois2)

    # Get mapping between sessions
    neuron_map = pfs.get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=batch_map_use)

    # Calculate orientation diff
    orient1_reg = orient1[neuron_map > -1]
    orient2_reg = orient2[neuron_map[neuron_map > -1]]
    orient_diff = orient1_reg - orient2_reg

    # Now make sure values range from -pi/2 to pi/2
    orient_diff[orient_diff < -np.pi/2] = orient_diff[orient_diff < -np.pi/2] + np.pi
    orient_diff[orient_diff > np.pi/2] = orient_diff[orient_diff > np.pi/2] - np.pi

    return orient_diff


def plot_ROI_orientation(roi, ax=None, zoom_buffer: int or None = 10):
    """Plot neuron ROI with orientation overlaid. Zoom in with buffer to zoom_buffer around roi"""
    if ax is None:
        _, ax = plt.subplots()

    # Detect ROI
    rprops = measure.regionprops((roi > 0).astype(int))
    assert len(rprops) == 1, "More/less than one ROI detected"

    # Calculate orientation line
    xmajor = rprops[0].centroid[1] + rprops[0].axis_major_length / 2 * np.array([-1, 1]) * np.cos(rprops[0].orientation)
    ymajor = rprops[0].centroid[0] + rprops[0].axis_major_length / 2 * np.array([-1, 1]) * np.sin(rprops[0].orientation)

    # Plot
    ax.imshow(roi)
    ax.plot(xmajor, ymajor, 'r-')

    # Zoom in
    if zoom_buffer is not None:
        ax.set_xlim(np.array([np.min(xmajor), np.max(xmajor)]) + np.array([-zoom_buffer, zoom_buffer]))
        ax.set_ylim(np.array([np.min(ymajor), np.max(ymajor)]) + np.array([-zoom_buffer, zoom_buffer]))
        ax.invert_yaxis()

    return ax


def load_proj(mouse, arena, day, type: str in ['min', 'max'], list_dir: str = master_directory):
    """Load min/max projection for a session"""
    make_session_list(list_dir)
    dir_use = get_dir(mouse, arena, day)
    if type == 'min':
        im = np.array(Image.open(path.join(dir_use, 'ICMovie_min_proj.tif')))
    elif type == 'max':
        im = np.array(Image.open(path.join(dir_use, 'ICMovie_max_proj.tif')))

    return im


def plot_ROIs(rois, bkgrd: np.ndarray or bool = True, color: str = 'r', ax=None, bkgrd_cmap='gray',
              bkgrd_rasterized=False):
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
        ax.imshow(bkgrd, cmap=bkgrd_cmap, rasterized=bkgrd_rasterized)

    # Detect edges and plot neurons
    for roi in rois:
        xedges, yedges = detect_roi_edges(roi)
        ax.plot(xedges, yedges, color=color)

    # Remove everything
    ax.axis('off')

    return ax


def plot_ROIs_bw_sessions(mouse, arena1, day1, arena2, day2, proj: str in ['min', 'max', 'custom'] = 'min',
                          custom_bkgrd: np.ndarray or None = None, ax=None, bkgrd_cmap='gray',
                          bkgrd_rasterized: bool = True, plot_all_rois: bool = True):
    """Plot ROIs from two sessions in different colors with co-active cells in green.

    Currently only works for same-day sessions - does not register ROIs between sessions due to
    affine transformation data being loaded from MATLAB into python improperly.

    Can provide a custom background with proj='custom' and backgroun=np.ndarray

    plot_all_rois: False = only plot correctly registered cell outlines, default = True (plot all)
    """

    # Load ROIs and projection from first session
    rois1 = load_ROIs(mouse, arena1, day1)
    if proj in ['min', 'max']:
        bkgrd = load_proj(mouse, arena1, day1, proj)
    elif proj == 'custom':
        bkgrd = custom_bkgrd

    # Load rois from second session
    rois2 = load_ROIs(mouse, arena2, day2)

    # Load map between two sessions
    neuron_map = pfs.get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=True)

    if ax is None:
        _, ax = plt.subplots()

    # Now plot ROIs
    try:
        _, rois2 = register_all_ROIs(mouse, arena1, day1, arena2, day2, method='all')
    except AssertionError:
        print("No _tform.csv file found for this pair of sessions. Check directory and create in MATLAB to run.")
        print("Plotting without transforming second session ROIs, be warned!")

    # Plot ROIs registered between sessions
    plot_ROIs(rois2[neuron_map[neuron_map >= 0]], bkgrd=bkgrd, color='r', ax=ax, bkgrd_cmap=bkgrd_cmap,
              bkgrd_rasterized=bkgrd_rasterized)
    plot_ROIs(rois1[neuron_map >= 0], bkgrd=False, color='r', ax=ax)  # Overlapping ROIs from sesh1

    # Plot ROIs from each session
    if plot_all_rois:
        plot_ROIs(rois1[neuron_map < 0], bkgrd=False, color='g', ax=ax)  # First session
        plot_ROIs(rois2, bkgrd=False, color='y', ax=ax) # Second session
        # plot_ROIs(rois1, bkgrd=bkgrd, color='g', ax=ax)  # First session
        # plot_ROIs(rois2, bkgrd=False, color='y', ax=ax)  # Second session

    return ax


def detect_roi_edges(roi_binary):
    """Detect roi edges and organize them nicely in CW/CCW fashion"""
    edges = feature.canny(roi_binary)  # detect edges
    inds = np.where(edges) # Get edge locations in pixels
    isort = np.argsort(np.arctan2(inds[1] - inds[1].mean(), inds[0] - inds[0].mean()))  # Sort by angle from center

    xedges = np.append(inds[1][isort], inds[1][isort[0]])
    yedges = np.append(inds[0][isort], inds[0][isort[0]])

    return xedges, yedges


def register_all_ROIs(mouse, arena1, day1, arena2, day2, method: str in ["combined", "all"]="combined"):

    # Load in ROIs
    rois1 = load_ROIs(mouse, arena1, day1)
    rois2 = load_ROIs(mouse, arena2, day2)

    # Combine into one 2d array
    rois1_comb = np.sum(rois1, axis=0)
    rois2_comb = np.sum(rois2, axis=0)

    # load in transform
    base_dir = Path(get_dir(mouse, arena1, day1))
    reg_sess_info = find_eraser_session(mouse, arena2, day2)
    reg_sess_info['Date'] = f"0{reg_sess_info['Date']}" if len(reg_sess_info['Date']) == 9 else reg_sess_info['Date']  # fix date string format

    tform_file_name = (f"RegistrationInfo-{reg_sess_info['Animal']}-{'_'.join(reg_sess_info['Date'].split('/'))}-"
                       f"session{reg_sess_info['Session']}_tform.csv")
    assert (base_dir / tform_file_name).exists(), "RegistrationInfo..._tform.csv file missing, save with csvwrite in MATLAB"
    tform = pd.read_csv(base_dir / tform_file_name, header=None)

    # Now flip around the x/y translation coordinates to convert from MATLAB affine2d function to Scipy.
    tform.iloc[2, :2] = -tform.iloc[2, 1::-1]

    if method == "combined":
        rois2_reg = ndi.affine_transform(rois2_comb, tform.values.T, order=1, output_shape=rois1_comb.shape)
        rois1 = rois1_comb
    elif method == "all":
        rois2_all = []
        for roi in rois2:
            roi2_reg = ndi.affine_transform(roi, tform.values.T, order=1, output_shape=rois1_comb.shape)
            rois2_all.append(roi2_reg)
        rois2_reg = np.array(rois2_all)

    return rois1, rois2_reg


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
    # calc_orientation_diff_bw_sessions('Marble07', 'Shock', -2, 'Shock', -1)
    # register_all_ROIs('Marble07', 'Shock', -1, 'Shock', 1)
    plot_ROIs_bw_sessions('Marble07', 'Shock', -1, 'Shock', 1, proj='min')
