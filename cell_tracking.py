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


def get_num_neurons(mouse, date, session):
    """
    :param mouse:
    :param arena:
    :param day:
    :param list_dir:
    :return: nneurons: # neurons in a session
    """

    dir_use = sd.find_session_directory(mouse, date, session)
    im_data_file = path.join(dir_use, 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file, variable_names='NumNeurons')
    # PSAbool = im_data['PSAbool']
    nneurons = im_data['NumNeurons'][0][0]

    return nneurons


if __name__ == '__main__':
    get_num_neurons('Marble12', 'Open', -2)

    pass