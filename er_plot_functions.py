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
                         list_dir='E:\Eraser\SessionDirectories'):
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




