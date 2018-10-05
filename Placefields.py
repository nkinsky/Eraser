# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 11:06:20 2018

@author: Nat Kinsky
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.io as sio
import scipy.ndimage as sim
from os import path
from session_directory import find_eraser_directory as get_dir
import er_plot_functions as er
from mouse_sessions import make_session_list
from plot_helper import ScrollPlot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos

# Might want these later
# import csv
# import pandas as pd


def placefields(mouse, arena, day, list_dir='E:\Eraser\SessionDirectories', cmperbin=1,
                lims_method='auto'):
    """
    Make placefields of each neuron. Ported over from Will Mau's/Dave Sullivan's MATLAB
    function
    :param mouse: mouse name to analyze
    :param arena: arena to analyze
    :param day: day to analyze
    :param list_dir: list alternate sessiondirectories location here
    :param cmperbin: 4 default
    :param lims_method: 'auto' (default) takes limits of data, 'manual' looks for arena_lims.csv
            file in the session directory which supplies [[xmin, ymin],[xmax, ymax]]
    :return:
    """

    make_session_list(list_dir)

    # Get position and time information for .csv file (later need to align to imaging)
    dir_use = get_dir(mouse, arena, day, list_dir)
    speed, pos, t_track, sr = get_speed(dir_use)
    t_track = t_track[0:-1]  # chop last time data point to match t_track match speed/pos length

    # Import imaging data
    im_data_file = path.join(dir_use + '\imaging', 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)
    PSAbool = im_data['PSAbool']
    nneurons, _ = np.shape(PSAbool)
    try:
        sr_image = im_data['SampleRate']
    except KeyError:
        sr_image = 20

    # Convert position to cm
    pix2cm = er.get_conv_factors(arena)  # get conversion to cm for the arena
    pos_cm = pos*pix2cm
    speed_cm = speed*pix2cm

    # Align imaging and position data
    pos_align, speed_align, PSAbool_align, time_interp  = \
        align_imaging_to_tracking(pos_cm, speed_cm, t_track, PSAbool, sr_image)

    # Smooth speed data for legitimate thresholding, get limits of data
    speed_sm = np.convolve(speed_align, np.ones(2*int(sr))/(2*sr), mode='same')  # smooth speed
    lims = [np.min(pos_cm, axis=1), np.max(pos_cm, axis=1)]
    good = np.ones(len(speed_sm)) == 1
    isrunning = good
    isrunning[speed_sm < 1.5] = False

    # Get the mouse's occupancy in each spatial bin
    occmap, runoccmap, xEdges, yEdges, xBin, yBin = \
        makeoccmap(pos_align, lims, good, isrunning, cmperbin)

    # Get rid of non-running epochs
    xrun = pos_align[0, isrunning]
    yrun = pos_align[1, isrunning]
    PSAboolrun = PSAbool_align[:, isrunning]
    xBinrun = xBin[isrunning]
    yBinrun = yBin[isrunning]
    nGood = len(xrun)

    # Construct place field and compute mutual information
    neurons = list(range(0, nneurons))
    tmap_us = []
    tcounts = []
    tmap_gauss = []
    for idn, neuron in enumerate(neurons):
        tmap_us_temp, tcounts_temp, tmap_gauss_temp = \
            makeplacefield(PSAboolrun[neuron, :], xrun, yrun, xEdges, yEdges, runoccmap,
                       cmperbin=cmperbin)
        tmap_us.append(tmap_us_temp)
        tcounts.append(tcounts_temp)
        tmap_gauss.append(tmap_gauss_temp)

    # Shuffle to get p-value!

    # save variables to working dir! as .pkl files?


    return occmap, runoccmap, xEdges, yEdges, xBin, yBin, tmap_us, tmap_gauss, tcounts, xrun, yrun, PSAboolrun


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
    pos = er.fix_pos(pos)  # fix any points at 0,0 by interpolating between closest good points
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
    if ax == None:
        _, ax = plt.subplots()

    cmap_use = mp.colors.Colormap(cmap)
    cmap_use.set_bad('white', 1.)
    ax.imshow(array, cmap=cmap_use)

    # Need to add something here and at the end to set back to original cmap just in case

    return ax


def align_imaging_to_tracking(pos_cm, speed_cm, time_tracking, PSAbool, sr_imaging):
    """
    Aligns imaging and tracking data and spits out all at the sample rate of the imaging data
    :param pos_cm:
    :param time_tracking:
    :param sr_imaging:
    :param PSAbool:
    :return: pos_align, speed_align, PSAbool_align, t_imaging - all aligned to t_imaging
    """

    # Get timestamps for PSAbool
    _, nframes = np.shape(PSAbool)
    t_imaging = np.arange(0, nframes/sr_imaging, 1/sr_imaging) + \
                np.min(time_tracking)  # tracking software starts image capture
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
    :return:
    """

    # number of frames and neurons
    nframes = np.sum(runoccmap)
    nneurons = PSAbool.shape(0)

    # get dwell map
    p_x = runoccmap.flatten()/nframes
    okpix = runoccmap.flatten() > 4  # only grab pixels occupied for at least 4 frames...
    p_x = p_x[okpix]  # only grab good pixels

    # get probability of spiking and not spiking
    p_k1 = np.sum(PSAbool,1)/nframes  # probability of spiking
    p_k0 = 1 - p_k1

    # Compute information metrics
    p_1x = []
    p_0x = []
    ipos = []
    mi = []
    isec = []
    ispk = []
    for neuron in np.arange(nneurons):
        # Get probability of spike given location, tmap, only taking good pixels
        p1xtemp = tmap_us.flatten()
        p1xtemp = p1xtemp[okpix]
        p_1x.append(p1xtemp)
        p0xtemp = 1 - p1xtemp
        p_0x.append(p0xtemp)

        # compute positional information for k=1 and k=0
        i_k1 = p1xtemp * np.log(p1xtemp / p_k1[neuron])
        i_k0 = p0xtemp * np.log(p0xtemp / p_k0[neuron])

        # sum these to make true positional information - NK follow up here to understand this
        ipostemp = i_k1 + i_k0
        ipos.append(ipostemp)

        # compute mutual information
        mi.append(np.nansum(p_x * ipostemp))

        # compute information content per second or event
        isec.append(np.nansum(p1xtemp * p_x * np.log2(p1xtemp / p_k1[neuron])))
        ispk.append(isec[neuron] * p_k1[neuron])
        

class PFobj:
    def __init__(self, tmap_us, tmap_gauss, x, y, PSAbool):
        self.tmap_us = tmap_us
        self.tmap_sm = tmap_gauss
        self.x = x
        self.y = y
        self.PSAbool = PSAbbool
        self.nneurons = PSAbool.shape[0]

    def pfscroll(self, current_position=0):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another

        :param current_position:
        :return:
        """

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) for n in range(self.nneurons)]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to make it through
        self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm),
                            current_position=current_position, n_frames=self.nneurons,
                            n_rows=1, n_cols=3, figsize=(17.2, 5.3), titles=titles,
                            x=self.x, y=self.y, PSAbool=self.PSAbool,
                            tmap_us=self.tmap_us, tmap_sm=self.tmap_sm)


if __name__ == '__main__':
    # placefields('Marble07', 'Open', -2, list_dir=r'C:\Eraser\SessionDirectories')
    occmap, runoccmap, xEdges, yEdges, xBin, yBin, tmap_us, tmap_gauss, \
    tcounts, xrun, yrun, PSAbool = placefields(
        'Marble07', 'Open', -2, list_dir=r'C:\Eraser\SessionDirectories')
    PFo = PFobj(tmap_us, tmap_gauss, xrun, yrun, PSAbool)
    PFo.pfscroll()
    pass
