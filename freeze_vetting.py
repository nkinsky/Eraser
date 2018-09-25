# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:59:20 2018

@author: Nat Kinsky
"""

import pandas as pd
from os import path
from glob import glob
import numpy as np
import er_plot_functions as er
import matplotlib.pyplot as plt
import csv

# list file appendices for each session and mouse names
ff_append = ['*1_EXPOSURE.csv', '*2_EXPOSURE.csv', '*4_4hr.csv',
             '*5_REEXPOSURE.csv', '*6_REEXPOSURE.csv', '*7_week.csv']
mouse_names = ['GENERAL_1', 'GENERAL_2', 'GENERAL_3', 'GENERAL_4']

## NRK update below to grab the appropriate file location for each computer
ff_dir = r'E:\Evan\0.25mA protocol\GEN_Pilots\FREEZING\GEN_1' # Evan's computer
# ff_dir = r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser\GEN_pilots\GEN_1'  # Nat's laptop

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


def plot_frz_comp(mouse_name, velocity_threshold=1.5, min_freeze_duration=10):
    """ Plots comparison between FF freezing and freezing calculated by us

    :param mouse_name:
    :return:
    """
    # get freezing by us
    fratio = er.get_all_freezing(mouse_name, day_des=[-2, -1, 4, 1, 2, 7], arenas=['Shock'],
                                 velocity_threshold=velocity_threshold, min_freeze_duration=min_freeze_duration,
                                 list_dir='E:\Eraser\SessionDirectories')

    # get freezing by freezeframe
    frz_avg, frz_by_min = get_ff_freezing(mouse_name)
    _, ax = plt.subplots()
    ax.scatter(frz_avg, fratio*100)
    ax.set_xlabel('Freezing by FF (%)')
    ax.set_ylabel('Freezing by us (%)')
    ax.set_title('Vel thresh = ?? min_freeze_dur = ??')

    # plot theoretically perfect match
    bounds = np.append(ax.get_ybound(), ax.get_xbound())
    bmin = np.min(bounds)
    bmax = np.max(bounds)
    ax.plot([bmin, bmax], [bmin, bmax], 'k--')


if __name__ == '__main__':
    get_ff_freezing('GENERAL_1')
    pass
