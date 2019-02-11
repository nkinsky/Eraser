# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:59:20 2019

@author: Nat Kinsky
"""

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.io as sio
import scipy.ndimage as sim
from os import path
import session_directory as sd
from session_directory import load_session_list
import er_plot_functions as er
from mouse_sessions import make_session_list
from plot_helper import ScrollPlot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos
# from progressbar import ProgressBar  # NK need a better version of this
from tqdm import tqdm
from pickle import dump, load


def get_num_neurons(mouse, date, session, er_arena=None, er_day=None):
    """Gets number of neurons in a given session.
    :param mouse:
    :param arena:
    :param day:
    :param list_dir:
    :return: nneurons: # neurons in a session
    """

    if er_arena is None:
        dir_use = sd.find_session_directory(mouse, date, session)
    else:
        dir_use = sd.find_eraser_directory(mouse, er_arena, er_day)
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file, variable_names='NumNeurons')
    # PSAbool = im_data['PSAbool']
    nneurons = im_data['NumNeurons'][0][0]

    return nneurons


def plot_num_neurons(nneurons, arena1='Shock', arena2='Open',
                     day_labels=['-2', '-1', '4hr', '1', '2', '7']):
    """
    Plots # neurons active in each arena on a given day. For eraser but could be used elsewhere
    :param nneurons: nmice x ndays x narenas array. arena1/2 = shock/open by default
    :return: fig, ax
    """

    nmice, ndays, narenas = nneurons.shape
    fig, ax = plt.subplots()
    ax.plot(np.matlib.repmat(np.arange(0, ndays), nmice, 1), nneurons[:, :, 0], 'bo')
    lineshock, = ax.plot(np.arange(0, ndays), np.nanmean(nneurons[:, :, 0], axis=0), 'b-')
    ax.plot(np.matlib.repmat(np.arange(0, ndays), nmice, 1), nneurons[:, :, 1], 'ro')
    lineopen, = ax.plot(np.arange(0, ndays), np.nanmean(nneurons[:, :, 1], axis=0), 'r-')
    plt.legend((lineshock, lineopen), (arena1, arena2))
    ax.set_ylabel('# Neurons')
    ax.set_xlabel('Day')
    ax.set_xticks(np.arange(0, ndays))
    ax.set_xticklabels(day_labels)

    return fig, ax


if __name__ == '__main__':
    get_num_neurons('Marble12', 'Open', -2)

    pass