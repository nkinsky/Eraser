# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 11:06:20 2018

@author: Nat Kinsky
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os import path
import skvideo.io
from glob import glob
from session_directory import find_eraser_directory as get_dir
# import pickle
from scipy.signal import decimate
import session_directory as sd
import eraser_reference as err
import scipy as sp
sd.make_session_list()  # update session list
plt.rcParams['pdf.fonttype'] = 42
import helpers as hlp

def display_frame(ax, vidfile):

    """
    For displaying the first frame of a video
    :param
        ax: matplotlib.pyplot axes
        vidfile: full path to video file
    :return:
    """
    vid = skvideo.io.vread(vidfile, num_frames=1)
    ax.imshow(vid[0])


def plot_trajectory(ax, posfile, xcorr=0, ycorr=0):
    """
    For plotting mouse trajectories.
    :param
        ax: axes to plot into
        posfile: location of mouse's csv position file
        xcorr, ycorr: x and y values to subtract from all values before plotting
    :return:
        ax: numpy axes!
    """
    pos = pd.read_csv(posfile, header=None)

    # adjust data if specified
    if xcorr != 0:
        pos.iloc[0, :] = pos.iloc[0, :] - xcorr
    if ycorr != 0:
        pos.iloc[1, :] = pos.iloc[1, :] - ycorr
    ax = pos.T.plot(0, 1, ax=ax, legend=False)

    return ax


def get_bad_epochs(mouse, arena, day):
    """
    Identifies bad epochs where mouse is at 0,0 for manual correction
    :param mouse:
    :param arena:
    :param day:
    :return:
    """

    # Commment here
    dir_use = get_dir(mouse, arena, day)

    # Comment here
    pos = get_pos(dir_use)

    # Comment here
    bad_bool = np.logical_and(pos[0, :] == 0, pos[1, :] == 0)

    # Comment here
    bad_epochs = get_freezing_epochs(bad_bool)

    # Insert code here to print bad epochs to screen if you wish. Might be easier in the long run
    #print(bad_epochs)
    return bad_epochs


def plot_frame_and_traj(ax, dir_use, plot_frame=True, xcorr=0, ycorr=0):

    """
    Plot mouse trajectory on top of the video frame
    :param
        ax: axes to plot into
        dir: directory housing the pos.csv and video tif file
        plot_frams: True = plot AVI frame, False = plot trajectory only
    :return:
    """
    pos_location = glob(path.join(dir_use + '\FreezeFrame', 'pos.csv'))
    avi_location = glob(path.join(dir_use + '\FreezeFrame', '*.avi'))

    try:
        if plot_frame:
            display_frame(ax, avi_location[0])
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
        plot_trajectory(ax, pos_location[0], xcorr=xcorr, ycorr=ycorr)
        if plot_frame:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
    except IndexError:  # plot just the trajectory if the avi file is missing
        plot_trajectory(ax, pos_location[0], xcorr=xcorr, ycorr=ycorr)


def plot_experiment_traj(mouse, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Open', 'Shock'],
                         disp_fratio=False):
    """
    Plot mouse trajectory for each session
    :param
        mouse: name of mouse
        day_des: days to plot (day 0 = shock day, day -2 = 2 days before shock, 4 = 4 hr after shock (special case))
        arenas: 'Open' and/or 'Shock'
        disp_fratio: true = display freezing ratio on plot
    :return: h: figure handle
    """
    nsesh = len(day_des)
    narena = len(arenas)
    fig, ax = plt.subplots(narena, nsesh, figsize=(12.7, 4.8), squeeze=False)

    # Iterate through all sessions and plot stuff
    for idd, day in enumerate(day_des):
        for ida, arena in enumerate(arenas):
            # try:
                dir_use = get_dir(mouse, arena, day)

                # Label stuff
                ax[ida, idd].set_xlabel(str(day))

                if idd == 0:
                    ax[ida, idd].set_ylabel(arena)
                if ida == 0 and idd == 0:
                    ax[ida, idd].set_title(mouse)

                axis_off(ax[ida, idd])
                plot_frame_and_traj(ax[ida, idd], dir_use)

                if disp_fratio:

                    velocity_threshold, min_freeze_duration, pix2cm = get_conv_factors(arena)
                    # if arena == 'Open':
                    #     # NK Note - velocity threshold is just a guess at this point
                    #     # Also need to ignore positions at 0,0 somehow and/or interpolate
                    #     velocity_threshold = 15
                    #     min_freeze_duration = 75
                    # elif arena == 'Shock':
                    #     velocity_threshold = 15
                    #     min_freeze_duration = 10

                    freezing, velocity = detect_freezing(dir_use,velocity_threshold=velocity_threshold,
                                               min_freeze_duration=min_freeze_duration, arena=arena,
                                               pix2cm=pix2cm)
                    fratio = freezing.sum()/freezing.__len__()
                    fratio_str = '%0.2f' % fratio # make it a string

                # Label stuff - hack here to make sure things get labeled if try statement fails during plotting
                ax[ida, idd].set_xlabel(str(day))
                if idd == 0:
                    ax[ida, idd].set_ylabel(arena)
                if ida == 0 and idd == 0:
                    ax[ida, idd].set_title(mouse)

                if idd == 0 and ida == 0 and disp_fratio:
                    ax[ida, idd].set_ylabel(fratio_str)
                elif disp_fratio:
                    ax[ida, idd].set_title(fratio_str)

            # except:
            #     print(['Error processing ' + mouse + ' ' + arena + ' ' + str(day)])

    return fig, ax


def plot_trajectory_overlay(mouse, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Open', 'Shock'],
                         xmin=False, ymin=False, xlim=False, ylim=False):
    """
    Plot mouse trajectory for each session
    :param
        mouse: name of mouse
        day_des: days to plot (day 0 = shock day, day -2 = 2 days before shock, 4 = 4 hr after shock (special case))
        arenas: 'Open' and/or 'Shock'
        xmin, ymin: #arenas by #days arrays or lists of minimum x/y values. False = don't adjust
        xlim, ylim: limits of x and y data you want to set as plotted. False = don't adjust.
    :return: h: figure handle
    """
    nsesh = len(day_des)
    narena = len(arenas)

    # Allocate xmin and ymin to zero if not designated
    if not xmin:
        xmin = np.zeros((narena, nsesh))
    if not ymin:
        ymin = np.zeros((narena, nsesh))

    # Cast xmin and ymin as 2d ndarrays for later
    xmin = np.reshape(xmin, (1, -1))
    ymin = np.reshape(ymin, (1, -1))

    fig, ax = plt.subplots(1, narena, figsize=(2.3, narena*2.3), squeeze=False)

    # Iterate through all sessions and plot stuff
    for ida, arena in enumerate(arenas):
        for idd, day in enumerate(day_des):
            dir_use = get_dir(mouse, arena, day)
            plot_frame_and_traj(ax[0, ida], dir_use, plot_frame=False, xcorr=xmin[ida, idd],
                                ycorr=ymin[ida, idd])
        # set x and y limits
        if xlim is not False:
            ax[0, ida].set_xlim(xlim)
        if ylim is not False:
            ax[0, ida].set_ylim(ylim)

    # resize figure
    fig.set_size_inches(6.2, 5.7)

    return fig, ax


def axis_off(ax):
    """
    Turn off all x and y axes and all tickmarks. Same as axis off command in MATLAB
    :param ax: axes handle
    :return:
    """
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off',  # labels along the bottom edge are off
        right='off',
        left='off',
        labelleft='off')


def detect_freezing(dir_use, velocity_threshold=1.5, min_freeze_duration=10, arena='Shock',
                    pix2cm=0.0969):
    # NK - need to update thresholds and min_freeze_duration above to reflect actual times/velocities,
    # NOT frames so that we can generalize between cages
    """
        Detect freezing epochs. input thresholds are in frames/sec and frames and should
        be calculated using get_conv_factors for freezeframe or cineplex videos. Defaults
        are set for freeze frame videos using Will's arbitrary ("by eye") method
        :param
            velocity_threshold: anything below this threshold is considered
                freezing. Frames/sec.
            min_freeze_duration: also, the epoch needs to be longer than
                this scalar. Frames.
                plot_freezing: logical, whether you want to see the results.
        :return:
            freezing: boolean of if mouse is freezing that frame or not
    """

    pos = get_pos(dir_use)
    pos, nbad = fix_pos(pos)
    print(dir_use + ': nbadpts = ' + str(nbad[0]) + ' max_in_a_row = ' + str(nbad[3]))
    # print(str(nbad[0]))  # for debugging
    # print('nbadpts = ' + str(nbad[0]))
    video_t = get_timestamps(dir_use)

    # Downsample Cineplex data to approximately match freezeframe acquisition rate
    # Lucky for us 30 Hz / 3.75 Hz = 8!!!
    if arena == 'Open':
        pos = decimate(pos, 8, axis=1, zero_phase=True, ftype='fir')
        video_t = decimate(video_t, 8, zero_phase=True, ftype='fir')

        # Add in an extra point to the end of the time stamp in the event they end up the same
        # length after downsampling - bugfix
        if pos.shape[1] == video_t.__len__():
            video_t = np.append(video_t, video_t[-1])

    try:
        pos = pos*pix2cm  # convert to centimeters
    except TypeError:
        pos = pos * pix2cm  # convert to centimeters
    pos_diff = np.diff(pos.T, axis=0)  # For calculating distance.
    time_diff = np.diff(video_t)  # Time difference.
    distance = np.hypot(pos_diff[:, 0], pos_diff[:, 1])  # Displacement.
    if (time_diff.__len__()) == (distance.__len__()):
        distance = distance[0:-1]
    velocity = np.concatenate(([0], distance // time_diff[0:-1]))  # Velocity. cm/sec (pixels/sec)
    freezing = velocity < velocity_threshold

    freezing_epochs = get_freezing_epochs(freezing)

    # Get duration of freezing in frames.
    freezing_duration = np.diff(freezing_epochs)

    # If any freezing epochs were less than ~3 seconds long (2.67 at SR = 3.75), get rid of
    # them.
    for this_epoch in freezing_epochs:
        if np.diff(this_epoch) < min_freeze_duration:
            freezing[this_epoch[0]:this_epoch[1]] = False

    return freezing, velocity


def get_pos(dir_use):
    """
    Open csv file and get position data
    :param
        dir_use: home directory
    :return
        pos: nd array of x and y position data for each timestamp
    """

    # Grab position either by directory or mouse/arena/day inputs

    try:  # look in either freezeframe directory or base directory
        pos_file = path.join(dir_use + '\FreezeFrame', 'pos.csv')
        temp = pd.read_csv(pos_file, header=None)
        pos = temp.as_matrix()
    except IOError:  # look in base directory if above is missing # FileNotFoundError is IOError in earlier versions
        pos_file = path.join(dir_use, 'pos.csv')
        temp = pd.read_csv(pos_file, header=None)
        pos = temp.as_matrix()

    return pos


def fix_pos(pos):
    """
    Fixes any points at (0,0) or (nan,nan) by interpolating between closest defined points
    :param pos: position data from get_pot
    :return: pos_fix: fixed position data
    :return: nbad: length 3 tuple (total # bad pts, # at start, # at end, max # in a row)
    """
    npts = pos.shape[1]
    zero_bool = np.bitwise_and(pos[0, :] == 0, pos[1,:] == 0)
    nan_bool = np.bitwise_and(np.isnan(pos[0, :]), np.isnan(pos[1, :]))
    bad_pts = np.where(np.bitwise_or(zero_bool, nan_bool))

    pos_fix = pos
    nbad_start = 0
    nbad_end = 0
    nbad_max = 1
    for pt in bad_pts[0]:

        # Increment/decrement until you find closest good point above/below bad point
        pt_p = [pt - 1, pt + 1]
        n = 0
        while pt_p[0] in bad_pts[0]:
            pt_p[0] -= 1
            n += 1
        while pt_p[1] in bad_pts[0]:
            pt_p[1] += 1
            n += 1

        if pt_p[0] < 0:  # use first good point if bad pt is at the beginning
            pos_fix[:, pt] = pos[:, pt_p[1]]
            nbad_start += 1
        elif pt_p[1] > npts:  # use last good point if bad pt is at the end
            pos_fix[:, pt] = pos[:, pt_p[0]]
            nbad_end += 1
        else:  # interpolate if good pts exist on either side
            x_p = pos[0, pt_p]
            y_p = pos[1, pt_p]

            pos_fix[0, pt] = np.interp(pt, pt_p, x_p)
            pos_fix[1, pt] = np.interp(pt, pt_p, y_p)
        nbad_max = np.max([nbad_max, n])

    nbad = (bad_pts[0].shape[0], nbad_start, nbad_end, nbad_max)

    return pos_fix, nbad


def get_timestamps(dir_use):
    """
    Get timestamps from Index csv file
    :param:
        dir_use: home directory
    :return:
        t: nd array of timestamps
    """
    try:
        time_file = glob(path.join(dir_use + '\FreezeFrame', '*Index*.csv'))
        temp = pd.read_csv(time_file[0], header=None)
    except IOError:  # FileNotFoundError is IOError in earlier versions
        time_file = glob(path.join(dir_use, '*Index*.csv'))
        temp = pd.read_csv(time_file[0], header=None)

    t = np.array(temp.iloc[:, 0])

    return t


def get_freezing_epochs(freezing):
    """
        returns indices of when freezing starts and stops.
    """
    padded_freezing = np.concatenate(([0], freezing, [0]))
    status_changes = np.abs(np.diff(padded_freezing))

    # Find where freezing begins and ends.
    freezing_epochs = np.where(status_changes == 1)[0].reshape(-1, 2)

    # NK - get a handle on below before you adjust it!!!
    freezing_epochs[freezing_epochs >= len(freezing)] = len(freezing) - 1

    # Only take the middle.
    # freezing_epochs = freezing_epochs[1:-1]

    return freezing_epochs


def get_all_freezing(mouse, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Open', 'Shock'],
                     list_dir='E:\Eraser\SessionDirectories', velocity_threshold=1.044, min_freeze_duration=10):
    """
    Gets freezing ratio for all experimental sessions for a given mouse.
    :param
        mouse: Mouse name (string), e.g. 'DVHPC_5' or 'Marble7'
        arenas: 'Open' (denotes square) or 'Shock' or 'Circle' (denotes open field circle arena)
        day_des: array of session days -2,-1,0,1,2,7 and 4 = 4hr session on day 0
        list_dir: alternate location of SessionDirectories
    :return:
        fratios: narena x nsession array of fratios
    """
    nsesh = len(day_des)
    narena = len(arenas)

    # Iterate through all sessions and get fratio
    fratios = np.ones((narena, nsesh))*float('NaN')  # pre-allocate fratio as nan
    for idd, day in enumerate(day_des):
        for ida, arena in enumerate(arenas):
            # print(mouse + " " + str(day) + " " + arena)
            try:

                pix2cm = get_conv_factors(arena)
                dir_use = get_dir(mouse, arena, day, list_dir=list_dir)
                freezing = detect_freezing(dir_use, velocity_threshold=velocity_threshold,
                                           min_freeze_duration=min_freeze_duration,
                                           arena=arena, pix2cm=pix2cm)[0]
                fratios[ida, idd] = freezing.sum()/freezing.__len__()
            except (IOError, IndexError, TypeError):  # FileNotFoundError is IOError in earlier versions
                # print(['Unknown error processing ' + mouse + ' ' + arena + ' ' + str(day)])
                print(['Unknown file missing and/or IndexError for ' + mouse + ' ' + arena + ' ' + str(day)])
                print('Freezing left as NaN for this session')

    return fratios


def plot_all_freezing(mice, days=[-2, -1, 4, 1, 2, 7], arenas=['Open', 'Shock'], velocity_threshold=1.0,
                      min_freeze_duration=10, title=''):

    """
    Plots freezing ratios for all mice
    :param mice: list of all mice to include in plot
        days
    :return: figure and axes handles, and all freezing values in freeze_ratio_all (narenas x ndays x nmice)
    """
    plot_colors = ['b', 'r']
    ndays = len(days)
    nmice = len(mice)
    narenas = len(arenas)

    fratio_all = np.empty((narenas, ndays, nmice))
    for idm, mouse in enumerate(mice):
        fratio_all[:, :, idm] = get_all_freezing(mouse, day_des=days, arenas=arenas,
                                        velocity_threshold=velocity_threshold, min_freeze_duration=min_freeze_duration)

    fig, ax = plt.subplots()
    # fratio_all = np.random.rand(2,7,5) # for debugging purposes

    # NK note - can make much of below into a general function to plot errorbars over a scatterplot in the future
    fmean = np.nanmean(fratio_all, axis=2)
    fstd = np.nanstd(fratio_all, axis=2)

    days_plot = np.asarray(list(range(ndays)))
    days_str = [str(e) for e in days]
    for ida, arena in enumerate(arenas):
        if arena is 'Open':
            offset = -0.05
        elif arena is 'Shock':
            offset = 0.05

        ax.errorbar(days_plot + offset, fmean[ida, :], yerr=fstd[ida, :], color=plot_colors[ida])

        for idm, mouse in enumerate(mice):
            fratio_plot = fratio_all[ida, :, idm]  # Grab only the appropriate mouse and day
            good_bool = ~np.isnan(fratio_plot)  # Grab only non-NaN values
            h = ax.scatter(days_plot[good_bool] + offset, fratio_plot[good_bool],
                           c=plot_colors[ida], alpha=0.2)

            # Hack to get figure handles for each separately - need to figure out how to put in iterable variable
            if arena == 'Open':
                hopen = h
            elif arena == 'Shock':
                hshock = h

    ax.set_xlim(days_plot[0]-0.5, days_plot[-1]+0.5)
    ax.set_xlabel('Session/Day')
    ax.set_ylabel('Freezing Ratio')
    if len(arenas) == 2:
        ax.legend((hopen, hshock), arenas)
    plt.xticks(days_plot, days_str)
    ax.set_title(title)

    return fig, ax, fratio_all


def get_conv_factors(arena, vthresh=1.45, min_dur=2.67):
    """
    Gets thresholds in cm/sec and num frames as well as pix2cm conversion factor for an arena
    Assumes 3.75 Hz framerate (Cineplex data must first be downsampled to 3.75 Hz)
    Better in future will be to do everything in seconds and cm/sec.
    :param
        arena: 'Open' (square open field), or 'Circle' (circle open field), or 'Shock'
        vthresh: velocity threshold in cm/sec, default = 1.45cm/sec
        min_dur: mouse must be below vthresh for this amount of time consecutively to
        be considered freezing, default = 2.67 sec
    :return:
        velocity thresh: counts as freezing when mouse is below this, cm/sec
        min_freeze_duration: mouse must be below velocity_thresh for this # frames currently in frames
        pix2cm: conversion factor from pixels to centimeters
    """

    if arena == 'Open':
        # Also need to ignore positions at 0,0 somehow and/or interpolate - not that many
        # Better yet would be to downsample to 4 frames/sec?
        # Probably need to filter these somehow - 5.5 cm/sec seems awfully fast for freezing
        # velocity_threshold = 5.4  # in pixels/sec, ~ 1.45cm/sec (3.7 would give us 1 cm/sec)
        # min_freeze_duration = 80  # 2.67 sec at SR = 30
        pix2cm = 0.27  # convert pixels to cm

        # SR = 30;  # frames/sec
    elif arena == 'Circle':
        # velocity_threshold = 7.2 # in pixels/sec, 5.0 would give us 1 cm/sec
        # min_freeze_duration = 80  # 2.67 sec at SR = 30
        pix2cm = 0.20  # convert pixels to cm

        # SR = 30  # frames/sec
    elif arena == 'Shock':
        # velocity_threshold = 15  # in pixels/sec ~ 1.45 cm/sec (10.3 would give us 1 cm/sec)
        # min_freeze_duration = 10  # 2.67 sec at SR = 3.75
        pix2cm = 0.0969  # convert pixels to cm
        # SR = 3.75  # frames/sec

    # velocity_threshold = vthresh/pix2cm  # in pixels/sec
    # velocity_threshold = 1.5  # cm/sec
    # min_freeze_duration = 10  # in frames at 3.75 frames/sec (~2.67 sec)
    # min_freeze_duration = np.round(min_dur*SR)

    return pix2cm


def write_all_freezing(fratio_all, filepath, days=[-2, -1, 4, 1, 2, 7]):
    """Writes freezing levels each day to a csv file

    :param fratio_all: 2 x 7 x nmice ndarray with freezing ratio values
    :param filepath: full file path to output csv file
    :param days: list of days to plot (note day 0 is by default missing)
    :return: nothing - writes to a csv file
    """

    # nmice = fratio_all.shape[2]

    # Construct day labels from input
    day_labels = []
    for day in days:
        if day != 4:
            day_labels.append('day ' + str(day))
        elif day == 4:
            day_labels.append(str(day) + ' hrs')

    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Open field', 'rows = mice', 'values = fratios'])
        writer.writerow(day_labels)
        writer.writerows(fratio_all[0, :, :].T)
        writer.writerow(['Shock Arena'])
        writer.writerows(fratio_all[1, :, :].T)


def plot_overlaps(overlaps, days=[-1, 4, 1, 2, 7]):
    """

    :param overlaps: nmice x 5sesh x narenas ndarray with cell overlap ratios
    relative to day -2. if multiple arenas dim 0 must be Shock v Shock and dim 1 must
    be Shock v Open
    :return: fig and ax handles
    """

    try:
        nmice, _, narenas = overlaps.shape
    except ValueError:
        nmice = 1
        _, narenas = overlaps.shape
    fig, ax = plt.subplots()
    if nmice != 1:
        ax.plot(np.matlib.repmat(np.arange(0, 5), nmice, 1), overlaps[:, :, 0], 'bo')
        lineshock, = ax.plot(np.arange(0, 5), np.nanmean(overlaps[:, :, 0], axis=0), 'b-')
    elif nmice == 1:
        lineshock, = ax.plot(np.arange(0, 5), overlaps[:, 0], 'bo-')
    ax.set_xlabel('Day/session')
    ax.set_ylabel('Overlap Ratio (Shock Day -2 = ref)')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([str(sesh) for sesh in days])

    if narenas == 2:
        if nmice != 1:
            ax.plot(np.matlib.repmat(np.arange(0, 5), nmice, 1), overlaps[:, :, 1], 'ro')
            linebw, = ax.plot([0, 1, 2, 3, 4], np.nanmean(overlaps[:, :, 1], axis=0), 'r-')
        elif nmice == 1:
            linebw, = ax.plot(np.arange(0, 5), overlaps[:, 1], 'ro-')

        ax.legend((lineshock, linebw), ('Shock v Shock', 'Shock v Open'))
    elif narenas == 1:
        ax.legend((lineshock,), ('Shock v Shock',))

    return fig, ax


def DIFreeze(mouse, days=[-2,-1,4,1,2,7]):
    fratios = get_all_freezing(mouse, days)
    frz_shock = fratios[1, :]
    frz_open = fratios[0, :]
    DIfrzing = []
    for (fo, fs) in zip(frz_open, frz_shock):
        DIfrz = [(fo-fs)/(fo+fs)]
        DIfrzing += DIfrz
    return DIfrzing


def DIhist(mice):
    import numpy as np
    import matplotlib.pyplot as plt
    DIarray = []
    for mouse in mice:
        # this should spit out the mouse with issues, then you can zero in after that via debugging
        # (make sure to comment out try/except statement first)
        try:
            DIval = DIFreeze(mouse)
            DIarray += DIval
        except ValueError:
            print('Error in ' + mouse)

    DIgood = []
    for id in DIarray:
        if np.isnan(id) == False:
               DIgood += [id]

    fig, ax = plt.subplots()
    ax.hist(DIgood)
    ax.set_xlabel('Discrim. Index')
    ax.set_ylabel('count')

def DIscatter(mice):
    # Importing files from matlab - look in placefields function in Placefields file, snippets here
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from os import path
    dir_use = 'E:\Evan\Figures\DI_Scatter'
    im_data_file = path.join(dir_use, 'DI_Neurons.mat')
    im_data = sio.loadmat(im_data_file)  # This imports ALL the variables in the .mat file listed above into im_data
    #create variable to hold neuron data
    matlab = im_data['maprate']
    #index DI_neurons to separate into specific mice
    DI_neurons = matlab['abmeanratechange']
    DI_neuron11 = DI_neurons[0,0:6]  # can get other variables stored in im_data other than PSAbool too
    DI_neuron12 = DI_neurons[0,6:12]
    DI_neuron07 = DI_neurons[0,12:18]
    DI_neuron14 = DI_neurons[0,18:24]
    DI_neuron19 = DI_neurons[0,24:30]
    DI_neuron24 = DI_neurons[0,30:36]
    DI_neuron25 = DI_neurons[0,36:42]
    DI_neuron27 = DI_neurons[0,42:48]
    DI_neuron29 = DI_neurons[0,48:54]
    DI_neuron17 = DI_neurons[0,54:60]
    DI_neuron20 = DI_neurons[0,60:66]
    DI_neuron06 = DI_neurons[0,66:]
    DIarray = []
    for mouse in mice:
        # this should spit out the mouse with issues, then you can zero in after that via debugging
        # (make sure to comment out try/except statement first)
        try:
            DIval = DIFreeze(mouse)
            DIarray += DIval
        except ValueError:
            print('Error in ' + mouse)
    DI_11 = DIarray[0:6]
    DI_12 = DIarray[6:12]
    DI_07 = DIarray[12:18]
    DI_14 = DIarray[18:24]
    DI_19 = DIarray[24:30]
    DI_24 = DIarray[30:36]
    DI_25 = DIarray[36:42]
    DI_27 = DIarray[42:48]
    DI_29 = DIarray[48:54]
    DI_17 = DIarray[54:60]
    DI_20 = DIarray[60:66]
    DI_06 = DIarray[66:]
    M11 = (DI_11, DI_neuron11)
    M12 = (DI_12, DI_neuron12)
    M07 = (DI_07, DI_neuron07)
    M14 = (DI_14, DI_neuron14)
    M19 = (DI_19, DI_neuron19)
    M24 = (DI_24, DI_neuron24)
    M25 = (DI_25, DI_neuron25)
    M27 = (DI_27, DI_neuron27)
    M29 = (DI_29, DI_neuron29)
    M17 = (DI_17, DI_neuron17)
    M20 = (DI_20, DI_neuron20)
    M06 = (DI_06, DI_neuron06)
    data = [M11, M12, M07, M14, M19, M24, M25, M27, M29, M17, M20, M06]
    groups = ["Marble11", "Marble12", "Marble07", "Marble14", "Marble19", "Marble24", "Marble25", "Marble27", "Marble29", "Marble17", "Marble20", "Marble06"]
    colors = ['red', 'blue', 'green', 'brown', 'black', 'purple', 'yellow', 'grey', 'magenta', 'cyan', 'orange','pink']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c = color, edgecolors='none', s=30, label=group)
    plt.legend(loc=2)
    plt.title("Discrimination Scatter Plot for all Marbles")
    plt.xlabel("DI_Freeze")
    plt.ylabel("Abs(DI_Neuron)")
    plt.show()


def on_or_off(mice):
    # Importing files from matlab - look in placefields function in Placefields file, snippets here
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from os import path
    dir_use = 'E:\Evan\Figures\DI_Scatter'
    im_data_file = path.join(dir_use, 'DI_Neurons.mat')
    im_data = sio.loadmat(im_data_file)  # This imports ALL the variables in the .mat file listed above into im_data
    # create variable to hold neuron data
    matlab = im_data['maprate']
    # index DI_neurons to separate into specific mice
    values_off = np.zeros((len(mice),6))
    values_on = np.zeros((len(mice),6))
    values_both = np.zeros((len(mice),6))
    y = -1
    for mouse in mice:
        y += 1
        cc_turn_on_total = 0
        cc_turn_off_total = 0
        cc_turn_both_total = 0
        for x in list(range(6)):
            cc_turn_on = 0
            cc_turn_off = 0
            cc_turn_both = 0
            for c in list(range(len(im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][:,0]))):
                if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 0] > 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 1] == 0:
                    cc_turn_off += 1
                    cc_turn_off_total += 1
                if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 0] == 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 1] > 0:
                    cc_turn_on += 1
                    cc_turn_on_total += 1
                if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 0] > 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][x][c, 1] > 0:
                    cc_turn_both += 1
                    cc_turn_both_total += 1
            values_off[y, x] = cc_turn_off
            values_on[y, x] = cc_turn_on
            values_both[y, x] = cc_turn_both
            print('The number of cells that turned on during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_on))
            print('The number of cells that turned off during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_off))
            print('The number of cells that turned on and off during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_both))
        print('The total of cells that turned on for ' + str(mouse) + ' is ' + str(cc_turn_on_total))
        print('The total of cells that turned off for ' + str(mouse) + ' is ' + str(cc_turn_off))
        print('The total of cells that turned on and off for ' + str(mouse) + ' is ' + str(cc_turn_both_total))
        print('The average of cells that turned on for ' + str(mouse) + ' is ' + str(cc_turn_on_total / 6))
        print('The average of cells that turned off for ' + str(mouse) + ' is ' + str(cc_turn_off_total / 6))
        print('The average of cells that turned both on and off for ' + str(mouse) + ' is ' + str(cc_turn_both / 6))
    return values_off, values_on, values_both

def on_or_off_single(mouse):
    # Importing files from matlab - look in placefields function in Placefields file, snippets here
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from os import path
    dir_use = 'E:\Evan\Figures\DI_Scatter'
    im_data_file = path.join(dir_use, 'DI_Neurons.mat')
    im_data = sio.loadmat(im_data_file)  # This imports ALL the variables in the .mat file listed above into im_data
    # create variable to hold neuron data
    matlab = im_data['maprate']
    # index DI_neurons to separate into specific mice
    values_off = np.zeros((1,3))
    values_on = np.zeros((1,3))
    values_both = np.zeros((1,3))
    cc_turn_on_total = 0
    cc_turn_off_total = 0
    cc_turn_both_total = 0
    for idx, x in enumerate(list(range(-2, 6))):
        cc_turn_on = 0
        cc_turn_off = 0
        cc_turn_both = 0
        for c in list(range(len(im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][:,0]))):
            if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 0] > 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 1] == 0:
                cc_turn_off += 1
                cc_turn_off_total += 1
            if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 0] == 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 1] > 0:
                cc_turn_on += 1
                cc_turn_on_total += 1
            if im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 0] > 0 and im_data['maprate'][im_data['maprate']['animal'] == mouse]['difrate'][idx][c, 1] > 0:
                cc_turn_both += 1
                cc_turn_both_total += 1
        values_off[0, idx] = cc_turn_off
        values_on[0, idx] = cc_turn_on
        values_both[0, idx] = cc_turn_both
        print('The number of cells that turned on during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_on))
        print('The number of cells that turned off during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_off))
        print('The number of cells that turned on and off during session ' + str(x) + ' for ' + str(mouse) + ' is ' + str(cc_turn_both))
    print('The total of cells that turned on for ' + str(mouse) + ' is ' + str(cc_turn_on_total))
    print('The total of cells that turned off for ' + str(mouse) + ' is ' + str(cc_turn_off))
    print('The total of cells that turned on and off for ' + str(mouse) + ' is ' + str(cc_turn_both_total))
    print('The average of cells that turned on for ' + str(mouse) + ' is ' + str(cc_turn_on_total / 6))
    print('The average of cells that turned off for ' + str(mouse) + ' is ' + str(cc_turn_off_total / 6))
    print('The average of cells that turned both on and off for ' + str(mouse) + ' is ' + str(cc_turn_both / 6))
    return values_off, values_on, values_both

# def DI_CC_scatter(mice):
#     #Importing files from matlab - look in placefields function in Placefields file, snippets here
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
#     for mouse in mice:
#         DIarray =[]
#          #this should spit out the mouse with issues, then you can zero in after that via debugging
#          #(make sure to comment out try/except statement first)
#         try:
#             DIval = DIFreeze(mouse)
#             DIarray += DIval
#             cc_off, cc_on, cc_both = on_or_off_single(mouse)
#             on_off_norm = np.divide((cc_on +  cc_off),(cc_on + cc_off + cc_both))
#             ax.scatter(DIarray, on_off_norm)
#         except ValueError:
#             print('Error in ' + mouse)
#     plt.title("Dan's Generalizers (need to reevaluate behavioral thresholds for G vs D")
#     plt.xlabel("DI_Freeze")
#     plt.ylabel("Normalized cell count for on and off cells")
#     plt.show()

def DI_CC_scatter_2(mice1,mice2):
    #Importing files from matlab - look in placefields function in Placefields file, snippets here
    nmice = len(mice1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    DIarray_total = []
    on_off_norm_total = np.empty((nmice, 3))
    id = 0
    for mouse1, mouse2 in zip(mice1, mice2):
        DIarray =[]
         #this should spit out the mouse with issues, then you can zero in after that via debugging
         #(make sure to comment out try/except statement first)
        try:
            DIval = DIFreeze(mouse1, days=[1, 2, 7])
            DIarray += DIval
            DIarray_total += DIval
            cc_off, cc_on, cc_both = on_or_off_single(mouse2)
            on_off_norm = np.divide((cc_on + cc_off), (cc_on + cc_off + cc_both))
            # on_off_norm_total += on_off_norm
            on_off_norm_total[id, :] = on_off_norm
            ax.scatter(DIarray, on_off_norm)
        except ValueError:
            print('Error in ' + mouse1)
        id += 1
    title = input("What is your title?")
    plt.title(title)
    plt.xlabel("DI_Freeze")
    plt.ylabel("Normalized cell count for on and off cells")
    # try:
    on_off_norm_total = on_off_norm_total.reshape(-1)
    DIarray_total = np.array(DIarray_total)
    good_pts_bool = np.bitwise_and(~np.isnan(DIarray_total), ~np.isnan(on_off_norm_total.reshape(-1)))
    z = np.polyfit(DIarray_total[good_pts_bool], on_off_norm_total[good_pts_bool], 1)
    p = np.poly1d(z)
    plt.plot(ax.get_xlim(), p(ax.get_xlim()), "r--")
     # except ValueError:
     #     print('Error in trend line')
    Corrcoef = sp.stats.pearsonr(DIarray_total[good_pts_bool], on_off_norm_total[good_pts_bool])
    plt.annotate("Pearson Correlation Coefficients: R = " + str(round(Corrcoef[0],3)) + "; P = " + str(round(Corrcoef[1],3)), xy=(0.15, 0.95), xycoords='axes fraction')
    plt.show()
    return Corrcoef

def sanitycheck(file):
 # Importing files from matlab - look in placefields function in Placefields file, snippets here
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from os import path
    dir_use = "E:\Evan\Figures\DI_Scatter"
    im_data_file = path.join(dir_use, file)
    im_data = sio.loadmat(im_data_file)  # This imports AL
    return im_data['ans']
    #hist = hlp.plot_prob_hist(im_data['ans'])


if __name__ == '__main__':
    import eraser_reference as err
    DI_CC_scatter_2(err.ani_mice_good_2, err.ani_mice_good_2)
    #on_or_off_single("Marble24")
    # test comment by evan
    #disc = sanitycheck("disc_score_raw.mat")


    pass