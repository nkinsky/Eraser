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
    pos_location = glob(path.join(dir, 'pos.csv'))
    avi_location = glob(path.join(dir, '*.avi'))

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
    narena = len(arena)
    fig, ax = plt.subplots(narena, nsesh)

    # Iterate through all sessions and plot stuff
    for idd, day in day_des:
        for ida, arena in arenas:
            try:
                dir_use = get_dir(mouse,arena,day,list_dir=list_dir)
                plot_frame_and_traj(ax[ida,idd],dir_use)
                # Label stuff
                if ida == 1:
                    ax[ida,idd].set_xlabel(str(day))
                if idd == 0:
                    ax[ida,idd].set_ylabel(arena)
                if ida == 0 and idd == 0:
                    ax[ida, idd].set_title(mouse)
            except:
                print(['Error processing ' arena + ' ' + str(day)])

    return fig




