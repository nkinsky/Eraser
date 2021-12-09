# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 11:06:20 2018

@author: Nat Kinsky
"""
import os

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.io as sio
import scipy.ndimage as sim
from os import path

import session_directory
from session_directory import find_eraser_directory as get_dir
from session_directory import load_session_list, master_directory, make_session_list
session_list = load_session_list()
import er_plot_functions as er
from mouse_sessions import make_session_list
from plot_helper import ScrollPlot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos, plot_psax, plot_psay
# from progressbar import ProgressBar  # NK need a better version of this
from tqdm import tqdm
from pickle import dump, load
from skimage.transform import resize as sk_resize

# Might want these later
# import csv
# import pandas as pd


def load_pf(mouse, arena, day, session_index=None, pf_file='placefields_cm1_manlims_1000shuf.pkl'):
    if session_index is None:
        dir_use = get_dir(mouse, arena, day)
    elif isinstance(session_index, int) and session_index >= 0:
        dir_use = session_list[session_index]["Location"]
    elif session_index == 'cwd':
        dir_use = os.getcwd()
    elif os.path.exists(session_index):
        dir_use = session_index

    position_path = path.join(dir_use, pf_file)
    with open(position_path, 'rb') as file:
        PF = load(file)

    if type(PF.sr_image) is not int:
        PF.sr_image = PF.sr_image.squeeze()  # Backwards compatibility fix

    return PF


def get_PV1(mouse, arena, day, speed_thresh=1.5, pf_file='placefields_cm1_manlims_1000shuf.pkl'):
    """
    Gets PV for each session with no spatial bins
    :param mouse:
    :param arena:
    :param day:
    :param speed_thresh: exclude data points where mouse's smoothed speed is below this (1.5cm/s default)
    :param pf_file: default = 'placefields_cm1_manlims_1000shuf.pkl'_
    :return: PV1: nneurons long 1-d np array of event rates for each neuron across the whole session
    """
    try:
        PF = load_pf(mouse, arena, day, pf_file=pf_file)
        # Speed threshold PSAbool
        PFthresh = PF.PSAbool_align[:, PF.speed_sm > speed_thresh]
        sr_image = PF.sr_image
    except FileNotFoundError:
        print('No placefields file found - creating PV1 from neural data only - NO SPEED THRESHOLDING')
        dir_use = get_dir(mouse, arena, day)
        im_data_file = path.join(dir_use, 'FinalOutput.mat')
        im_data = sio.loadmat(im_data_file)
        PSAbool = im_data['PSAbool']
        PFthresh = PSAbool
        sr_image = im_data['SampleRate'][0]

    # Calculate PV
    nframes = PFthresh.shape[1]
    try:
        PV1d = PFthresh.sum(axis=1)/nframes * sr_image[0]
    except TypeError:  # Catch a few errors for mice where sr_image is not properly formatted
        PV1d = PFthresh.sum(axis=1) / nframes * sr_image
    return PV1d


def get_PV2(mouse, arena, day, speed_thresh=1.5, pf_file='placefields_cm1_manlims_1000shuf.pkl', rot_deg=0,
            resize_dims=None):
    """Gets a 2-d population vector of activity.  Basically just stack up all placefield maps.
    :param mouse:
    :param arena:
    :param day:
    :param speed_thresh:
    :param pf_file:
    :param resize_dims: dimensions to resize PV2 to (e.g. for doing across arena comparisons)
    :return: PV an nneurons x # spatial bins ndarray of calcium event activity
    """
    try:
        PF = load_pf(mouse, arena, day, pf_file=pf_file)
        # Speed threshold PSAbool
        if speed_thresh != PF.speed_thresh:  # Re-run placefields if using a different speed threshold
            print('Speed threshold for 2-d PVs does not match previous run - re-calculating placefields')
            PF = placefields(mouse, arena, day, speed_thresh=speed_thresh)

        # Rotate maps
        if rot_deg == 0:
            tmap_sm_rot = np.asarray(PF.tmap_sm)
            tmap_us_rot = np.asarray(PF.tmap_us)
        elif rot_deg in [90, 180, 270]:
            tmap_sm_rot = np.asarray(rotate_tmaps(PF.tmap_sm, rot_deg))
            tmap_us_rot = np.asarray(rotate_tmaps(PF.tmap_us, rot_deg))

        # Next, reshape tmaps from session 2 if doing across arena comparisons
        if rot_deg == 0 and resize_dims is None:
            tmap_us_rs, tmap_sm_rs = tmap_us_rot, tmap_sm_rot
        elif rot_deg == 0 and resize_dims is not None:
            tmap_us_rs = rescale_tmaps(tmap_us_rot, resize_dims)
            tmap_sm_rs = rescale_tmaps(tmap_sm_rot, resize_dims)
        elif rot_deg in [90, 180, 270] or rot_deg == 0 and resize_dims is not None:
            tmap_us_rs = rescale_tmaps(tmap_us_rot, resize_dims)
            tmap_sm_rs = rescale_tmaps(tmap_sm_rot, resize_dims)

        # Finally flatten maps into a 1d array.
        PVsm = np.asarray(tmap_sm_rs).reshape(PF.nneurons, -1)
        PVus = np.asarray(tmap_us_rs).reshape(PF.nneurons, -1)

    except FileNotFoundError:
        print('No placefields file found - can''t create 2d population vector')
        PVus, PVsm = np.nan, np.nan

    return PVus, PVsm


def rotate_tmaps(tmaps, rot_deg):
    """
        Rotate all transient maps in tmaps in 90 degree increments
        :param tmaps: list of tmaps
        :param rot_deg: int, # degrees to rotate map, must be 90 degree increments
        :return: list of rotated tmaps
        """
    rot = int(rot_deg / 90)
    tmaps_rot = []
    for tmap in tmaps:
        tmaps_rot.append(np.rot90(tmap, rot))

    return tmaps_rot


def rescale_tmaps(tmaps, new_size):
    """
    Resize all transient maps in tmaps to roughly match the shape specified in new_size
    :param tmaps: list of tmaps
    :param new_size: shape = (2,) list or tuple with shape to rescale to
    :return: list of reshaped tmaps
    """
    tmaps_rescale = []
    for tmap in tmaps:
            tmaps_rescale.append(sk_resize(tmap, new_size, anti_aliasing=True))

    return tmaps_rescale


def placefields(mouse, arena, day, cmperbin=1, nshuf=1000, speed_thresh=1.5, half=None,
                lims_method='auto', save_file='placefields_cm1.pkl', list_dir=master_directory,
                align_from_end=False, keep_shuffled=False, isrunning_custom=None):
    """
    Make placefields of each neuron. Ported over from Will Mau's/Dave Sullivan's MATLAB
    function
    :param mouse: mouse name to analyze
    :param arena: arena to analyze
    :param day: day to analyze
    :param cmperbin: 4 default
    :param lims_method: 'auto' (default) takes limits of data, 'file' looks for arena_lims.csv
            file in the session directory which supplies [[xmin, ymin],[xmax, ymax]], or you
            you can enter in [[xmin, ymin], [xmax, ymax]] manually
    :param nshuf: number of shuffles to perform for determining significance
    :param speed_thresh: speed threshold in cm/s
    :param save_file: default = 'placefields_cm1.pkl'. None = do not save
    :param align_from_end: False (default) align data assuming start of neural/behavioral data acquisition was
    synchronized, True = use end time-points to align (in case of bad triggering at beginning but good at end).
    :param half: None (default) = run whole session, 1 = run 1st half only, 2 = run 2nd half only, (odd/even not yet
    implemented as of 2021_02_01).
    :param keep_shuffled: True = keep shuffled smoothed tmaps (saved as .tmap_sm_shuf). False = default.
    :param isrunning_custom: use your own frames, specified in a boolean created by get_running_bool and modified by you
    as you please, to calculate the place field map for an arbitrary period of the session.
    :return:
    """

    make_session_list(list_dir)

    # Get position and time information for .csv file (later need to align to imaging)
    dir_use = get_dir(mouse, arena, day)
    speed, pos, t_track, sr = get_speed(dir_use)
    t_track = t_track[0:-1]  # chop last time data point to match t_track match speed/pos length

    # Display warning if "aligntoend" is in folder name but you are running with align_from_end=False
    if str(dir_use).upper().find('ALIGNTOEND') != -1 and align_from_end is False:
        print('Folder structure for ' + mouse + ' ' + arena + ': Day ' + str(day) + ' suggests you should align data from end of recording')
        print('RE-RUN WITH align_from_end=False!!!')

    # Import imaging data
    # im_data_file = path.join(dir_use + '\imaging', 'FinalOutput.mat')
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)
    PSAbool = im_data['PSAbool']
    nneurons, _ = np.shape(PSAbool)
    try:
        sr_image = im_data['SampleRate'].squeeze()
    except KeyError:
        sr_image = 20

    # Convert position to cm
    pix2cm = er.get_conv_factors(arena)  # get conversion to cm for the arena
    pos_cm = pos*pix2cm
    speed_cm = speed*pix2cm

    # Align imaging and position data
    pos_align, speed_align, PSAbool_align, time_interp = \
        align_imaging_to_tracking(pos_cm, speed_cm, t_track, PSAbool, sr_image, align_from_end=align_from_end)

    # Smooth speed data for legitimate thresholding, get limits of data
    speed_sm = np.convolve(speed_align, np.ones(2*int(sr))/(2*sr), mode='same')  # smooth speed

    # Get data limits
    if lims_method == 'auto':  # automatic by default
        lims = [np.min(pos_cm, axis=1), np.max(pos_cm, axis=1)]
    elif lims_method == 'file':  # file not yet enabled
        print('lims_method=''file'' not enabled yet')
    else:  # grab array!
        lims=lims_method

    good = np.ones(len(speed_sm)) == 1
    isrunning = good.copy()
    isrunning[speed_sm < speed_thresh] = False

    # Break up session into halves if necessary.
    if half is not None:
        # Identify # minutes and when half occurs
        half_id = np.floor(len(isrunning) / 2).astype('int')
        nminutes = np.ceil(len(isrunning) / sr_image / 60).astype(int)

        # Now chop things up!
        if half == 1:
            isrunning[half_id:] = False
        elif half == 2:
            isrunning[:half_id] = False
        elif half in ('odd', 'even'):
            odd_even_bool = np.zeros_like(isrunning, dtype=bool)
            start_minute = np.where([half == epoch for epoch in ['odd', 'even']])[0][0]
            for a in range(start_minute, nminutes, 2):
                odd_even_bool[a * 60 * sr_image:(a + 1) * 60 * sr_image] = 1
            isrunning[~odd_even_bool] = False

    # Use custom period for calculating placefields if specified
    if isrunning_custom is not None:
        isrunning = isrunning_custom

    # Get the mouse's occupancy in each spatial bin
    occmap, runoccmap, xEdges, yEdges, xBin, yBin = \
        makeoccmap(pos_align, lims, good, isrunning, cmperbin)

    # Get rid of non-running epochs
    xrun = pos_align[0, isrunning]
    yrun = pos_align[1, isrunning]
    PSAboolrun = PSAbool_align[:, isrunning]

    nGood = len(xrun)

    # Construct place field and compute mutual information
    neurons = list(range(0, nneurons))
    tmap_us, tcounts, tmap_gauss = [], [], []
    for idn, neuron in enumerate(neurons):
        tmap_us_temp, tcounts_temp, tmap_gauss_temp = \
            makeplacefield(PSAboolrun[neuron, :], xrun, yrun, xEdges, yEdges, runoccmap,
                       cmperbin=cmperbin)
        tmap_us.append(tmap_us_temp)
        tcounts.append(tcounts_temp)
        tmap_gauss.append(tmap_gauss_temp)

    # calculate mutual information
    mi, _, _, _, _ = spatinfo(tmap_us, runoccmap, PSAboolrun)

    # Shuffle to get p-value!
    pval, tmap_sm_shuf = [], []
    print('Shuffling to get placefield p-values')
    for neuron in tqdm(np.arange(nneurons)):
        rtmap, rtmap_sm = [], []
        shifts = np.random.randint(0, nGood, nshuf)
        for ns in np.arange(nshuf):
            # circularly shift PSAbool to disassociate transients from mouse location
            shuffled = np.roll(PSAboolrun[neuron, :], shifts[ns])
            map_temp, _, sm_map_temp = makeplacefield(shuffled, xrun, yrun, xEdges, yEdges, runoccmap,
                                      cmperbin=cmperbin)
            rtmap.append(map_temp)
            rtmap_sm.append(sm_map_temp)

        # Calculate mutual information of randomized vectors
        rmi, _, _, _, _ = spatinfo(rtmap, runoccmap, repmat(PSAboolrun[neuron, :], nshuf, 1))

        # Calculate p-value
        pval.append(1 - np.sum(mi[neuron] > rmi) / nshuf)

        # Aggregate shuffled maps if specified (worried this might kill memory)
        if keep_shuffled:
            tmap_sm_shuf.append(rtmap_sm)

    # save variables to working dirs as .pkl files in PFobject
    PFobj = PlaceFieldObject(tmap_us, tmap_gauss, xrun, yrun, PSAboolrun, occmap, runoccmap,
                 xEdges, yEdges, xBin, yBin, tcounts, pval, mi, pos_align, PSAbool_align,
                 speed_sm, isrunning, cmperbin, speed_thresh, mouse, arena, day, list_dir,
                             nshuf, sr_image, tmap_sm_shuf)

    if save_file is not None:
        PFobj.save_data(filename=save_file)

    return PFobj


def get_running_bool(mouse, arena, day,  speed_thresh=1.5, lims_method='auto', list_dir=master_directory,
                align_from_end=False):
    """Generate a boolean of times the animal is running to use with placefields. Can modify and then feed into
    placefields function to generate place field maps from arbitrary times throughout the session
    :param all same as in placefields function above
    :return isrunning: boolean when the mouse is running. """
    make_session_list(list_dir)

    # Get position and time information for .csv file (later need to align to imaging)
    dir_use = get_dir(mouse, arena, day)
    speed, pos, t_track, sr = get_speed(dir_use)
    t_track = t_track[0:-1]  # chop last time data point to match t_track match speed/pos length

    # Display warning if "aligntoend" is in folder name but you are running with align_from_end=False
    if str(dir_use).upper().find('ALIGNTOEND') != -1 and align_from_end is False:
        print('Folder structure for ' + mouse + ' ' + arena + ': Day ' + str(
            day) + ' suggests you should align data from end of recording')
        print('RE-RUN WITH align_from_end=False!!!')

    # Import imaging data
    # im_data_file = path.join(dir_use + '\imaging', 'FinalOutput.mat')
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)
    PSAbool = im_data['PSAbool']
    nneurons, _ = np.shape(PSAbool)
    try:
        sr_image = im_data['SampleRate'].squeeze()
    except KeyError:
        sr_image = 20

    # Convert position to cm
    pix2cm = er.get_conv_factors(arena)  # get conversion to cm for the arena
    pos_cm = pos * pix2cm
    speed_cm = speed * pix2cm

    # Align imaging and position data
    pos_align, speed_align, PSAbool_align, time_interp = \
        align_imaging_to_tracking(pos_cm, speed_cm, t_track, PSAbool, sr_image, align_from_end=align_from_end)

    # Smooth speed data for legitimate thresholding, get limits of data
    speed_sm = np.convolve(speed_align, np.ones(2 * int(sr)) / (2 * sr), mode='same')  # smooth speed

    # Get data limits
    if lims_method == 'auto':  # automatic by default
        lims = [np.min(pos_cm, axis=1), np.max(pos_cm, axis=1)]
    elif lims_method == 'file':  # file not yet enabled
        print('lims_method=''file'' not enabled yet')
    else:  # grab array!
        lims = lims_method

    good = np.ones(len(speed_sm)) == 1
    isrunning = good
    isrunning[speed_sm < speed_thresh] = False

    return isrunning


def makeoccmap(pos_cm, lims, good, isrunning, cmperbin):
    """
    Make Occupancy map for mouse
    :param pos_cm:
    :param lims:
    :param good:
    :param isrunning:
    :param cmperbin:
    :return: occmap, runoccmap, xEdges, yEdges, xBin, yBin
    """

    # Extract Limits
    xmin = lims[0][0]
    xmax = lims[1][0]
    ymin = lims[0][1]
    ymax = lims[1][1]

    # Make edges for hist2
    xrange = xmax - xmin
    yrange = ymax - ymin

    nXBins = np.ceil(xrange/cmperbin)
    nYBins = np.ceil(yrange/cmperbin)

    xEdges = np.linspace(0, nXBins, nXBins)*cmperbin + xmin
    yEdges = np.linspace(0, nYBins, nYBins)*cmperbin + ymin

    # Run 2d histogram function to get occupancy and running occupancy maps
    occmap, _, _ = np.histogram2d(pos_cm[0, good], pos_cm[1, good], bins=[xEdges, yEdges])
    runoccmap, _, _ = np.histogram2d(pos_cm[0, good & isrunning], pos_cm[1, good & isrunning],
                                     bins=[xEdges, yEdges])
    xBin = np.digitize(pos_cm[0, :], xEdges)
    yBin = np.digitize(pos_cm[1, :], yEdges)

    # rotate maps 90 degrees to match trajectory
    occmap = np.rot90(occmap, 1)
    runoccmap = np.rot90(runoccmap, 1)

    return occmap, runoccmap, xEdges, yEdges, xBin, yBin


def get_speed(dir_use):
    """

    :param dir_use: home directory for the mouse.

    :return: speed and pos (pix/sec, and pix), timestamps for tracking data (s),
     and sr (sample rate, Hz)
    """

    # Get position and time information for .csv file (later need to align to imaging)
    pos = er.get_pos(dir_use)  # pos in pixels
    pos, _ = er.fix_pos(pos)  # fix any points at 0,0 by interpolating between closest good points
    t = er.get_timestamps(dir_use)  # time in seconds

    # Calculate speed
    pos_diff = np.diff(pos.T, axis=0)  # For calculating distance.
    time_diff = np.diff(t)  # Time difference.
    distance = np.hypot(pos_diff[:, 0], pos_diff[:, 1])  # Displacement.
    speed = np.concatenate(([0], distance // time_diff[0:-1]))  # Velocity. cm/sec
    sr = np.round(1 / np.mean(time_diff))

    return speed, pos, t, sr


def imshow_nan(array, ax=None, cmap='viridis'):
    """
    Plot an array with nan values set as white.
    Not even sure this is necessary anymore - default might plot nans as white...
    :param array: array to plot - nans will be white
    :param ax: optional, if left out or set to None creates a new figure
    :param cmap: optional, default = viridis
    :return: ax
    """

    # Define plotting axes if not specified
    if ax is None:
        _, ax = plt.subplots()

    cmap_use = mp.colors.Colormap(cmap)
    cmap_use.set_bad('white', 1.)
    ax.imshow(array, cmap=cmap_use)

    # Need to add something here and at the end to set back to original cmap just in case

    return ax


def align_imaging_to_tracking(pos_cm, speed_cm, time_tracking, PSAbool, sr_imaging, align_from_end=False):
    """
    Aligns all tracking data to imaging data and spits out all at the sample rate of the imaging data. Note that this
    assumes you have aligned imaging start or end to tracking software externally.
    :param pos_cm:
    :param time_tracking:
    :param sr_imaging:
    :param PSAbool:
    :param align_from_end: set to True to use last frames of imaging/tracking software for alignment.  Use to cross-validate
    normal alignment (from start) or in case of faulty triggering at start. Default = False.
    :return: pos_align, speed_align, PSAbool_align, t_imaging - all aligned to t_imaging
    """

    # Get timestamps for PSAbool
    _, nframes = np.shape(PSAbool)
    if not align_from_end:
        t_imaging = np.arange(0, nframes/sr_imaging, 1/sr_imaging) + \
            np.min(time_tracking)  # tracking software starts image capture
    elif align_from_end:
        t_imaging = np.arange(-nframes/sr_imaging, 0, 1/sr_imaging) + \
            np.max(time_tracking)  # assume tracking software off ends image capture
    pos_align = np.empty((2, t_imaging.shape[0]))
    pos_align[0, :] = np.interp(t_imaging, time_tracking, pos_cm[0, :])
    pos_align[1, :] = np.interp(t_imaging, time_tracking, pos_cm[1, :])
    speed_align = np.interp(t_imaging, time_tracking, speed_cm)

    # Chop any frames in imaging data that extend beyond tracking data
    t_im_include_bool = np.max(time_tracking) > t_imaging
    PSAbool_align = PSAbool[:, t_im_include_bool]
    pos_align = pos_align[:, t_im_include_bool]
    speed_align = speed_align[t_im_include_bool]

    return pos_align, speed_align, PSAbool_align, t_imaging


def makeplacefield(PSAbool, x, y, xEdges, yEdges, runoccmap, cmperbin=4, gauss_std=2.5):
    """
    Make placefields from aligned imaging/tracking data
    :param PSAbool:
    :param x:
    :param y:
    :param xEdges:
    :param yEdges:
    :param runoccmap:
    :param cmperbin:
    :param smooth:
    :param gauss_std:
    :return:
    """
    # Get counts of where mouse was for each calcium event in PSAbool
    tcounts, _, _ = np.histogram2d(x[PSAbool==1], y[PSAbool==1], bins=[xEdges, yEdges])
    tsum = np.sum(tcounts)

    # rotate tcounts 90 degrees to match mouse trajectory when plotting later
    tcounts = np.rot90(tcounts, 1)

    # Normalize it
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore warnings due to NaNs produced from dividing by zero
        tmap_us = np.divide(tcounts, runoccmap)
    tmap_us[np.isnan(tmap_us)] = 0

    if tsum != 0:
        # smooth
        tmap_sm = sim.filters.gaussian_filter(tmap_us, sigma=gauss_std)  # NK need to vet this
        tmap_sm = tmap_sm*tsum/np.sum(tmap_sm)
    elif tsum == 0:  # edge case
        tmap_sm = tmap_us

    tmap_sm[runoccmap == 0] = np.nan
    tmap_us[runoccmap == 0] = np.nan

    return tmap_us, tcounts, tmap_sm


def spatinfo(tmap_us, runoccmap, PSAbool):
    """
    Calculates the Shannon mutual information I(X,K) between the random
    variables spike count [0,1] and position via the equations:

    (1) I_pos(xi) = sum[k>=0](P_k|xi * log(P_k|xi / P_k))

    (2) MI = sum[i=1->N](P_xi * I_pos(xi)

    where:
       P_xi is the probability the mouse is in pixel xi,
       RunOccMap./sum(RunOccMap(:)

       P_k is the probability of observing k spikes,
       sum(FT(neuron,:),2)/size(FT,2)

       P_k|xi is the conditional probability of observing k spikes in
       pixel xi, TMap_unsmoothed

   Ported over from Will Mau's MATLAB function

    :param tmap_us:
    :param runoccmap:
    :param PSAbool:
    :return: mi, isec, ispk, ipos, okpix
    """

    # number of frames and neurons
    nframes = np.sum(runoccmap)
    nneurons = PSAbool.shape[0]

    # get dwell map
    p_x = runoccmap.flatten()/nframes
    okpix = runoccmap.flatten() > 4  # only grab pixels occupied for at least 4 frames...
    p_x = p_x[okpix]  # only grab good pixels

    # get probability of spiking and not spiking
    p_k1 = np.sum(PSAbool, 1)/nframes  # probability of spiking
    p_k0 = 1 - p_k1  # probability of NOT spiking

    # Compute information metrics
    p_1x = []
    p_0x = []
    ipos = []
    mi = []
    isec = []
    ispk = []
    for neuron in np.arange(nneurons):
        # Get probability of spike given location, tmap, only taking good pixels
        try:
            p1xtemp = tmap_us[neuron].flatten()
        except (IndexError, AttributeError):
            p1xtemp = tmap_us[neuron].flatten()
        p1xtemp = p1xtemp[okpix]
        # p_1x.append(p1xtemp)  # NRK not sure why these were here in the first place - not necessary to keep!
        p0xtemp = 1 - p1xtemp
        # p_0x.append(p0xtemp)

        # compute positional information for k=1 and k=0
        # NRK Note - lots of RunTimeWarnings due to the arrays being mostly zero. Look into Olypher - is this ok? Must be!
        i_k1 = p1xtemp * np.log(p1xtemp / p_k1[neuron])  # Lots of nans - why?
        i_k0 = p0xtemp * np.log(p0xtemp / p_k0[neuron])  # All good values

        # sum these to make true positional information - NK follow up here to understand this
        ipostemp = i_k1 + i_k0
        ipos.append(ipostemp)

        # compute mutual information
        mi.append(np.nansum(p_x * ipostemp))

        # compute information content per second or event
        isec.append(np.nansum(p1xtemp * p_x * np.log2(p1xtemp / p_k1[neuron])))
        ispk.append(isec[neuron] * p_k1[neuron])

    return mi, isec, ispk, ipos, okpix


def load_all_mi(mice, arenas=['Open', 'Shock'], days=[-2, -1, 4, 1, 2, 7], pf_file='placefields_cm1_manlims_1000shuf.pkl'):
    """
    Get previously calculated mutual information for all mice/days/arenas
    :param mice: list
    :param arenas: list
    :param days: list
    :param pf_file: str
    :return: mimean_all: nmice x narenas x ndays nd-array of mean mutual info values
    """
    nmice = len(mice)
    ndays = len(days)
    mimean_all = np.ones((nmice, 2, ndays))*np.nan
    for idm, mouse in enumerate(mice):
        for ida, arena in enumerate(arenas):
            for idd, day in enumerate(days):
                try:
                    PFobj = load_pf(mouse, arena, day, pf_file=pf_file)
                    mimean_all[idm, ida, idd] = np.nanmean(PFobj.mi)
                except FileNotFoundError:
                    print('Missing file for ' + mouse + ' ' + arena + ' day ' + str(day))

    return mimean_all


def get_im_sample_rate(mouse, arena, day):
    """Gets sample rate for imaging data"""
    dir_use = get_dir(mouse, arena, day)
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file, variable_names='SampleRate')
    try:
        sr_image = im_data['SampleRate'].squeeze()
    except KeyError:
        sr_image = 20

    return sr_image


def remake_occmap(xBin, yBin, runoccmap, good_bool: None or np.ndarray or list = None):
    """Fix occmap generated by Placefields.placefields() prior to 11/29/2021 - bug results in occmap = runoccmap"""
    nx, ny = runoccmap.shape
    if isinstance(good_bool, (np.ndarray, list)):
        x_use, y_use = xBin[good_bool], yBin[good_bool]
    elif good_bool is None:
        x_use, y_use = xBin, yBin
    xEdges, yEdges = np.arange(0, nx + 1), np.arange(0, ny + 1)
    occmap, _, _ = np.histogram2d(x_use, y_use, bins=[xEdges + 0.5, yEdges + 0.5])
    occmap = np.rot90(occmap, 1)
    return occmap


class PlaceFieldObject:
    def __init__(self, tmap_us, tmap_gauss, xrun, yrun, PSAboolrun, occmap, runoccmap,
                 xEdges, yEdges, xBin, yBin, tcounts, pval, mi, pos_align, PSAbool_align,
                 speed_sm, isrunning, cmperbin, speed_thresh, mouse, arena, day,
                 list_dir, nshuf, sr_image, tmap_sm_shuf):
        self.tmap_us = tmap_us
        self.tmap_sm = tmap_gauss
        self.xrun = xrun
        self.yrun = yrun
        self.PSAboolrun = PSAboolrun
        self.nneurons = PSAboolrun.shape[0]
        self.occmap = occmap
        self.runoccmap = runoccmap
        self.xEdges = xEdges
        self.yEdges = yEdges
        self.xBin = xBin
        self.yBin = yBin
        self.tcounts = tcounts
        self.pval = pval
        self.mi = mi
        self.pos_align = pos_align
        self.PSAbool_align = PSAbool_align
        self.speed_sm = speed_sm
        self.isrunning = isrunning
        self.cmperbin = cmperbin
        self.speed_thresh = speed_thresh
        self.mouse = mouse
        self.arena = arena
        self.day = day
        self.list_dir = list_dir
        self.nshuf = nshuf
        self.sr_image = sr_image
        self.tmap_sm_shuf = tmap_sm_shuf

    def save_data(self, filename='placefields_cm1.pkl'):
        dir_use = get_dir(self.mouse, self.arena, self.day, self.list_dir)
        save_file = path.join(dir_use, filename)

        with open(save_file, 'wb') as output:
            dump(self, output)

    def pfscroll(self, current_position=0, pval_thresh=1.01, plot_xy=False, link_PFO=None):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another

        :param current_position: index in spatially tuned neuron ndarray to start with (clunky, since you don't
        know how many spatially tuned neurons you have until you threshold them below).
        :param pval_thresh: default = 1. Only scroll through neurons with pval (based on mutual information scores
        calculated after circularly permuting calcium traces/events) < pval_thresh
        :param plot_xy: plot x and y position versus time with calcium activity indicated in red.
        :param link_PFO: placefield object to link to for matched scrolling.
        :return:
        """

        # Get only spatially tuned neurons: those with mutual spatial information pval < pval_thresh
        if self.nshuf > 0:
            spatial_neurons = np.where([a < pval_thresh for a in self.pval])[0]
        elif self.nshuf == 0:
            spatial_neurons = np.arange(0, self.nneurons, 1)

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) for n in spatial_neurons]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to scroll through each neuron
        lims = [[self.xEdges.min(), self.xEdges.max()], [self.yEdges.min(), self.yEdges.max()]]
        if not plot_xy:
            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm),
                                current_position=current_position, n_neurons=len(spatial_neurons),
                                n_rows=1, n_cols=3, figsize=(17.2, 5.3), titles=titles,
                                x=self.pos_align[0, self.isrunning], y=self.pos_align[1, self.isrunning],
                                traj_lims=lims, PSAbool=self.PSAboolrun[spatial_neurons, :],
                                tmap_us=[self.tmap_us[a] for a in spatial_neurons],
                                tmap_sm=[self.tmap_sm[a] for a in spatial_neurons],
                                mouse=self.mouse, arena=self.arena, day=self.day, link_obj=link_PFO)
        elif plot_xy:
            # quick bugfix - earlier versions of Placefields spit out sr_image as multi-level list
            sr_image = self.sr_image
            while type(sr_image) is not int:
                sr_image = sr_image[0]

            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm, plot_psax, plot_psay),
                                current_position=current_position, n_neurons=len(spatial_neurons),
                                n_rows=3, n_cols=3, combine_rows=[1, 2], figsize=(12.43, 9.82), titles=titles,
                                x=self.pos_align[0, self.isrunning], y=self.pos_align[1, self.isrunning],
                                traj_lims=lims, PSAbool=self.PSAboolrun[spatial_neurons, :],
                                tmap_us=[self.tmap_us[a] for a in spatial_neurons],
                                tmap_sm=[self.tmap_sm[a] for a in spatial_neurons],
                                mouse=self.mouse, arena=self.arena, day=self.day, sample_rate=sr_image,
                                link_obj=link_PFO)


if __name__ == '__main__':
    placefields('Marble07', 'Open', -2)
    pass
