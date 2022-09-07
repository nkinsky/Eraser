# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:59:20 2019

@author: Nat Kinsky
"""

import numpy as np
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

# Make text save as whole words
plt.rcParams['pdf.fonttype'] = 42


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


def get_group_num_neurons(mice, days=[-2, -1, 4, 1, 2, 7], arenas=['Shock', 'Open']):
    """Gets # neurons for all mice/days/arenas specified

    :param mice: list
    :param days: list
    :param arenas: list
    :return: nneurons_all: nmice x ndays x narenas ndarray
    """
    nneurons = np.ones((len(mice), len(arenas), len(days))) * np.nan
    for idm, mouse in enumerate(mice):
        for ida, arena in enumerate(arenas):
            for idd, day in enumerate(days):
                try:
                    nneurons[idm, ida, idd] = get_num_neurons(mouse, '', '',
                                                er_arena=arena, er_day=day)
                except TypeError:
                    print('Missing neural data file for ' + mouse + ' Day ' + str(day) + ' ' + arena)

    return nneurons


def plot_num_neurons(nneurons, arena1='Shock', arena2='Open', day_labels=('-2', '-1', '4hr', '1', '2', '7'),
                     normalize=False, jitter=(-0.05, 0.05), colors = ('b', 'r'), ax=None, **kwargs):
    """
    Plots # neurons active in each arena on a given day. For eraser but could be used elsewhere
    :param nneurons: nmice x ndays x narenas array. arena1/2 = shock/open by default
    :param arena1/2: arena labels default1/2 = 'Shock'/'Open'.
    :param normalize: boolean False(default) = do not normalize, otherwise takes a
        string = day to normalize to, e.g. '-1' or '7'
    :param kwargs: inputs to matplotlib.pyplot.plot
    :return: fig, ax
    """

    nmice, narenas, ndays = nneurons.shape

    jitter = (jitter,) if isinstance(jitter, float) else jitter
    assert len(jitter) == narenas, 'Must enter jitter amount for each arena'
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # normalize nneurons to day indicated
    if normalize is not False:
        norm_sesh_ind = [day_labels.index(i) for i in day_labels if normalize in i][0]
        nneurons = norm_num_neurons(nneurons, norm_sesh_ind)

    ax.plot(np.matlib.repmat(np.arange(0, ndays), nmice, 1) + jitter[0], nneurons[:, 0, :],
            color=colors[0], marker='o', linestyle='None', **kwargs)
    lineshock, = ax.plot(np.arange(0, ndays) + jitter[0], np.nanmean(nneurons[:, 0, :], axis=0),
                         color=colors[0], marker=None, linestyle='-', **kwargs)
    if narenas == 2:
        ax.plot(np.matlib.repmat(np.arange(0, ndays), nmice, 1) + jitter[1], nneurons[:, 1, :],
                color=colors[1], marker='o', linestyle='None', **kwargs)
        lineopen, = ax.plot(np.arange(0, ndays) + jitter[1], np.nanmean(nneurons[:, 1, :], axis=0),
                            color=colors[1], marker=None, linestyle='-', **kwargs)
        plt.legend((lineshock, lineopen), (arena1, arena2))

    if normalize is False:
        ax.set_ylabel('# Neurons')
    else:
        ax.set_ylabel('Norm. # Neurons ' + normalize + '=ref')
    ax.set_xlabel('Day')
    ax.set_xticks(np.arange(0, ndays))
    ax.set_xticklabels(day_labels)

    return fig, ax


def norm_num_neurons(nneurons, norm_sesh_ind):
    """Normalize nneurons to a particular session

    :param nneurons: nmice x ndays x narenas array of active neurons
    :param norm_sesh_ind: index in dimension 1 (days) to normalize to. Default = 0.
    :return: nnorm:
    """
    nmice, narenas, ndays = nneurons.shape
    nnorm = nneurons.reshape(narenas * nmice, ndays) / \
            nneurons[:, :, norm_sesh_ind].reshape(narenas * nmice, -1)
    nnorm = nnorm.reshape((nmice, narenas, ndays))

    ### Old code here where I had ndays in dim 1. above should be better.
    # nnorm = nneurons.swapaxes(1, 2).reshape(narenas * nmice, ndays) / \
    #            nneurons[:, norm_sesh_ind, :].reshape(narenas * nmice, -1).copy()
    # nnorm = nnorm.reshape((nmice, narenas, ndays)).swapaxes(2, 1)

    return nnorm


if __name__ == '__main__':
    get_num_neurons('Marble12', 'Open', -2)

    pass