# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:59:20 2018

@author: Nat Kinsky
"""

import pandas as pd
from os import path, environ
from glob import glob
import numpy as np
import er_plot_functions as er
import matplotlib.pyplot as plt
import csv

# list file appendices for each session and mouse names
ff_append = ['*1_EXPOSURE.csv', '*2_EXPOSURE.csv', '*4_4hr.csv',
             '*5_REEXPOSURE.csv', '*6_REEXPOSURE.csv', '*7_week.csv']
mouse_names = ['GENERAL_1', 'GENERAL_2', 'GENERAL_3', 'GENERAL_4']

# NRK update below to grab the appropriate file location for each computer
if environ['COMPUTERNAME'] == 'CAS-2CUMM202-02':
    ff_dir = r'E:\Evan\0.25mA protocol\GEN_Pilots\FREEZING\GEN_1'  # Evan's computer
    list_dir = r'E:\Eraser\SessionDirectories'
elif environ['COMPUTERNAME'] == 'NATLAPTOP':
    ff_dir = r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser\GEN_pilots\GEN_1'  # Nat's laptop
    list_dir = r'C:\Eraser\SessionDirectories'
elif environ['COMPUTERNAME'] == 'NORVAL':
    ff_dir = r'E:\Eraser\GEN_pilots\GEN_1'
    list_dir = r'E:\Eraser\SessionDirectories'

ff_paths = [None]*len(ff_append)
for ida, names in enumerate(ff_append):
    ff_paths[ida] = glob(path.join(ff_dir, names))


def get_ff_freezing(mouse_name):
    """ Pulls freezing from FreezeFrame for all 6 sessions (excluding shock session) for the mouse specified

    returns: frz_avg, frz_by_min
    frz_avg: average freezing for the session
    frz_by_min: freezing for each minute of the session
    """

    # Find which mouse you entered
    mouse_bool = []
    for mouse in mouse_names:
        mouse_bool.append(mouse_name == mouse)

    # Loop through and get all freezing data for each 60 sec interval + avg freezing
    frz_avg_l = [None]*len(ff_paths)
    frz_by_min = np.empty((0, 10))
    for idf, ff_path in enumerate(ff_paths):
        ff_freeze_data = pd.read_csv(ff_path[0])
        mouse_min_frz = ff_freeze_data.iloc[int(np.where(mouse_bool)[0])+2, 1:11]
        frz_avg_l[idf] = mouse_min_frz.mean(axis=0)
        frz_by_min = np.append(frz_by_min, np.reshape(mouse_min_frz.values, [1, 10]), axis=0)

    frz_avg = np.asarray(frz_avg_l)

    return frz_avg, frz_by_min


def plot_frz_comp(mouse_name, velocity_threshold=1.5, min_freeze_duration=10, ax=None):
    """

    :param mouse_name: self explanatory
    :param velocity_threshold: considered freezing if mouse is below this, cm/s (1.5 = default)
    :param min_freeze_duration: considered freezing only if => this # frames (10 = default at 3.75 fps)
    :param ax: axes to plot into, default=None-> create new figure/axes
    :return: ax: axes handle to freezing comparison plot
    :return: fratio: freezing ratio calculated by us
    :return: ff_frz_avg: freezing % calculated by FreezeFrame
    """
    # get freezing by us
    fratio = er.get_all_freezing(mouse_name, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Shock'],
                                 velocity_threshold=velocity_threshold, min_freeze_duration=min_freeze_duration,
                                 list_dir=list_dir)
    fratio = np.reshape(fratio, fratio.shape[1])  # resize to make 1d

    # get freezing by freezeframe
    ff_frz_avg, ff_frz_by_min = get_ff_freezing(mouse_name)

    # create a new figure/axes if none are specified
    if ax is None:
        _, ax = plt.subplots()

    # Plot everything
    ax.scatter(ff_frz_avg, fratio*100)
    ax.set_xlabel('Freezing by FF (%)')
    ax.set_ylabel('Freezing by us (%)')
    ax.set_title('Vel thresh = ' + str(velocity_threshold) +
                 'cm/s, min_freeze_dur = ' + str(min_freeze_duration) + ' frames')

    # plot theoretically perfect match
    bounds = np.append(ax.get_ybound(), ax.get_xbound())
    bmin = np.min(bounds)
    bmax = np.max(bounds)
    xlim = ax.get_xlim()
    ax.plot([bmin, bmax], [bmin, bmax], 'k--')

    # get correlation coeff
    rmat = np.corrcoef(ff_frz_avg, fratio*100)
    rtxt = "%0.2f" % rmat[0, 1]  # make into a string

    # get linear correlation
    a = np.polyfit(ff_frz_avg, fratio*100, 1)

    # Plot linear regression line and put corr. coeff. on plot
    ax.plot(np.asarray(xlim), np.asarray(xlim) * a[0] + a[1], 'r-.')
    ax.set_xlim(xlim)
    ax.text(10, 40, r'$r^2$ = ' + rtxt)

    return ax, fratio, ff_frz_avg


def param_sweep_plot(mouse_name, vel_thresh=[0.25, 1, 1.5], frz_dur_thresh=[5, 10, 15]):
    """Does a parameter sweep of velocity and min_freeze thresholds to see how well
    our freezing values match up with FreezeFrame

    :param mouse_name: self-explanatory
    :param vel_thresh: velocities to sweep through
    :param frz_dur_thresh: minimum freeze durations in frames to sweep through
    :return: ax: plot axes
    """

    _, ax = plt.subplots(len(vel_thresh), len(frz_dur_thresh))

    for idv, vthresh_use in enumerate(vel_thresh):
        for idf, fthresh_use in enumerate(frz_dur_thresh):
            plot_frz_comp(mouse_name, velocity_threshold=vthresh_use,
                          min_freeze_duration=fthresh_use, ax=ax[idv, idf])

    return ax


if __name__ == '__main__':
    param_sweep_plot('GENERAL_1')
    pass
