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
import pickle

def display_frame(ax,vidfile):
    """
    For displaying the first frame of a video
    :param
        ax: matplotlib.pyplot axes
        vidfile: full path to video file
    :return:
    """
    vid = skvideo.io.vread(vidfile,num_frames=1)
    ax.imshow(vid[0])


def plot_trajectory(ax,posfile):
    """
    For plotting mouse trajectories.
    :param
        pos: nparray of x/y mouse location values
    :return:
    """
    pos = pd.read_csv(posfile,header=None)
    pos.T.plot(0,1,ax=ax,legend=False)

def plot_frame_and_traj(ax,dir):
    """
    Plot mouse trajectory on top of the video frame
    :param
        dir: directory housing the pos.csv and video tif file
    :return:
    """
    pos_location = glob(path.join(dir + '\FreezeFrame','pos.csv'))
    avi_location = glob(path.join(dir + '\FreezeFrame', '*.avi'))

    display_frame(ax,avi_location[0])
    plot_trajectory(ax,pos_location[0])

def plot_experiment_traj(mouse, day_des=[-2,-1,0,4,1,2,7], arenas=['Open','Shock'],
                         list_dir='E:\Eraser\SessionDirectories', disp_fratio = False):
    """
    Plot mouse trajectory for each session
    :param
        mouse: name of mouse
        day_des: days to plot (day 0 = shock day, day -2 = 2 days before shock, 4 = 4 hr after shock (special case))
        arenas: 'Open' and/or 'Shock'
    :return: h: figure handle
    """
    nsesh = len(day_des)
    narena = len(arenas)
    fig, ax = plt.subplots(narena, nsesh, figsize=(12.7,4.8))

    # Iterate through all sessions and plot stuff
    for idd, day in enumerate(day_des):
        for ida, arena in enumerate(arenas):
            try:
                dir_use = get_dir(mouse,arena,day,list_dir=list_dir)

                # Label stuff
                ax[ida,idd].set_xlabel(str(day))
                if idd == 0:
                    ax[ida,idd].set_ylabel(arena)
                if ida == 0 and idd == 0:
                    ax[ida, idd].set_title(mouse)

                axis_off(ax[ida,idd])
                plot_frame_and_traj(ax[ida,idd],dir_use)

                if disp_fratio:
                    # un-pickle your variables
                    pkl_file = path.join(dir_use,'Position.pkl')
                    FObj = pickle.load(open(pkl_file,'rb'))
                    fratio = FObj.freezing.sum()/len(FObj.freezing)
                    fratio_str = '%0.2f' % fratio # make it a string

                # Label stuff - hack here to make sure things get labeled regardless of plotting or not
                ax[ida, idd].set_xlabel(str(day))
                if idd == 0:
                    ax[ida, idd].set_ylabel(arena)
                if ida == 0 and idd == 0:
                    ax[ida, idd].set_title(mouse)

            except:
                print(['Error processing ' + arena + ' ' + str(day)])

    return fig

def axis_off(ax):
    """
    Turn off x and y axes and tickmarks
    :param ax: axes handle
    :return:
    """
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        right='off',
        left='off',
        labelleft='off')


def detect_freezing(dir_use, velocity_threshold=15, min_freeze_duration=10):
    # NK - need to update thresholds and min_freeze_duration above to reflect actual times/velocities,
    # NOT frames so that we can generalize between cages
    """
        Detect freezing epochs.
        :param
            velocity_threshold: anything below this threshold is considered
                freezing. Default set for freezeframe by Will's arbitrary method,
                likely NOT good for Cineplex data
            min_freeze_duration: also, the epoch needs to be longer than
                this scalar. Set to approximately 3 seconds.
                plot_freezing: logical, whether you want to see the results.
        :return:
            freezing: boolean of if mouse if freezing that frame or not
    """


    pos = get_pos(dir_use)
    video_t = get_timestamps(dir_use)
    pos_diff = np.diff(pos, axis=0)  # For calculating distance.
    time_diff = np.diff(video_t)  # Time difference.
    distance = np.hypot(pos_diff[:, 0], pos_diff[:, 1])  # Displacement.
    velocity = np.concatenate(([0], distance // time_diff))  # Velocity.
    freezing = self.velocity < velocity_threshold

    freezing_epochs = get_freezing_epochs(velocity)

    # Get duration of freezing in frames.
    freezing_duration = np.diff(freezing_epochs)

    # If any freezing epochs were less than ~3 seconds long, get rid of
    # them.
    for this_epoch in freezing_epochs:
        if np.diff(this_epoch) < min_freeze_duration:
            self.freezing[this_epoch[0]:this_epoch[1]] = False

    return freezing


def get_pos(dir_use):
    """
    Open csv file and get position data
    :param
        dir_use: home directory
    :return
        pos: nd array of x and y position data for each timestamp
    """
    pos_file = path.join(dir_use + '\FreezeFrame', 'pos.csv')
    temp = pd.read_csv(pos_file, header=None)
    pos = temp.as_matrix()

    return pos


def get_timestamps(dir_use):
    """
    Get timestamps from Index csv file
    :param:
        dir_use: home directory
    :return:
        t: nd array of timestamps
    """
    time_file = path.join(dir_use + '\FreezeFrame', '*Index.csv')
    temp = pd.read_csv(time_file, header=None)
    t = np.array(temp.iloc[:, 0])

    return t


def get_freezing_epochs(freezing):
    padded_freezing = np.concatenate(([0], freezing, [0]))
    status_changes = np.abs(np.diff(padded_freezing))

    # Find where freezing begins and ends.
    freezing_epochs = np.where(status_changes == 1)[0].reshape(-1, 2)

    # NK - get a handle on below before you adjust it!!!
    freezing_epochs[freezing_epochs >= len(freezing)] = len(freezing) - 1

    # Only take the middle.
    freezing_epochs = freezing_epochs[1:-1]

    return freezing_epochs