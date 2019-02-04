# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 09:53:00 2019

@author: Nat Kinsky
"""
import numpy as np
import scipy.io as sio
import scipy.stats as sstats
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os import path
import skvideo.io
from glob import glob
from session_directory import find_eraser_directory as get_dir
import session_directory as sd
# import pickle
from scipy.signal import decimate
import Placefields as pf
from mouse_sessions import make_session_list
import cell_tracking as ct


def get_neuronmap(mouse, arena1, day1, arena2, day2):
    """
    Get mapping between registered neurons from arena1/day1 to arena2/day2
    :param mouse:
    :param arena1: session 1 day/arena
    :param day1:
    :param arena2: session 2 day/arena
    :param day2:
    :return: neuron_map: an array the length of the number of neurons in session1. zeros indicate
    that neuron has no matched counterpart in session2.
    """

    make_session_list()  # Initialize session list

    # Identify map file
    dir_use = get_dir(mouse, arena1, day1)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    reg_filename = 'neuron_map-' + mouse + '-' + sd.fix_slash_date(reg_session['Date']) + \
                   '-session' + reg_session['Session'] + '.mat'
    map_file = path.join(dir_use, reg_filename)

    # Load file in
    map_data = sio.loadmat(map_file)
    map_import = map_data['neuron_map']['neuron_id'][0][0]  # Grab terribly formatted neuron map from matlab

    # Fix the map - spit out an array!
    neuron_map = fix_neuronmap(map_import)

    good_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)

    return neuron_map


def fix_neuronmap(map_import):
    """
    Fixes neuronmap input imported from matlab in cell format to spit out an array and converts
    to python nomenclature (e.g. first entry = 0)
    :param map_import: poorly formatted map imported from Nat's matlab output
    :return: map_fixed: a fixed map
    """

    map_fixed = np.ones_like(map_import)*np.nan  #pre-allocate to NaNs
    for idn, neuron in enumerate(map_import):
        if neuron[0] != 0:
            map_fixed[idn] = neuron[0][0]-1  # subtract 1 to convert to python numbering!

    return map_fixed


def classify_cells(neuron_map, reg_session, overlap_thresh=0.5):
    """
    Classifies cells as good, silent, and new.
    :param neuron_map:
    :param overlap_thresh: not functional yet. default (eventually) = consider new/silent if 0.5 overlap or less
    :return: good_map_bool, silent_ind, new_ind
    """

    # Get silent neurons
    silent_ind, _ = np.where(np.isnan(neuron_map))

    # Get new neurons
    nneurons2 = ct.get_num_neurons(reg_session['Animal'], reg_session['Date'],
                                   reg_session['Session'])

    new_ind, _ = np.where(np.ismember(np.arange(0,nneurons2), neuron_map))

def get_overlap(mouse, arena1, day1, arena2, day2):
    """
    Gets overlap of cells between sessions.
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :return: overlap_ratio1, overlap_ratio2, overlap_ratio_both:
    #overlapping cells between sessions divided by the
    the number of cells active in 1st/2nd/both sessions
    """
    neuron_map  = get_neuronmap(mouse, arena1, day1, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map)

    num_active1 = len(good_map_bool)
    num_active2 = sum(good_map_bool) + len(new_ind)
    num_active_both = len(good_map_bool) + len(new_ind)
    num_overlap = sum(good_map_bool)

    overlap_ratio1 = num_overlap/num_active1
    overlap_ratio2 = num_overlap/num_active2
    overlap_ratio_both = num_overlap/num_active_both

    return overlap_ratio1, overlap_ratio2, overlap_ratio_both


def pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2):
    """
    Gets placefield correlations between sessions.
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :return:
    """

    # Get mapping between sessions
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map)

    # load in placefield objects between sessions
    PF1 = pf.load_pf(mouse, arena1, day1, pf_file='placefields_cm1_manlims.pkl')
    PF2 = pf.load_pf(mouse, arena2, day2, pf_file='placefields_cm1_manlims.pkl')

    # Identify neurons with proper mapping between sessions
    good_map_ind = np.where(good_map_bool)
    ngood = len(good_map_ind)

    corrs_us = np.ndarray(ngood)  # Initialize correlation arrays
    corrs_sm = np.ndarray(ngood)
    # Step through each mapped neuron and get corrs between each
    for idn, neuron in enumerate(good_map_ind):
        corr_us, p_us = sstats.spearmanr(np.reshape(PF1.tmap_us, -1), np.reshape(PF2.tmap_us, -1))
        corr_sm, p_sm = sstats.spearmanr(np.reshape(PF1.tmap_sm, -1), np.reshape(PF2.tmap_sm, -1))
        corrs_us[idn] = corr_us
        corrs_sm[idn] = corr_sm

    return corrs_us, corrs_sm


if __name__ == '__main__':

    neuron_map = get_neuronmap('Marble11', 'Shock', -2, 'Shock', -1)
    classify_cells(neuron_map)
    pass