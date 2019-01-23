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
from session_directory import find_eraser_directory as get_dir
from session_directory import load_session_list
import er_plot_functions as er
from mouse_sessions import make_session_list
from plot_helper import ScrollPlot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos
# from progressbar import ProgressBar  # NK need a better version of this
from tqdm import tqdm
from pickle import dump, load


def get_num_neurons(mouse, arena, day, list_dir='E:\Eraser\SessionDirectories'):
    """
    :param mouse:
    :param arena:
    :param day:
    :param list_dir:
    :return: nneurons: # neurons in a session
    """

    dir_use = get_dir(mouse, arena, day, list_dir)
    im_data_file = path.join(dir_use + '\imaging', 'FinalOutput.mat')
    im_data = sio.loadmat(im_data_file)
    PSAbool = im_data['PSAbool']
    nneurons, _ = np.shape(PSAbool)

    return nneurons