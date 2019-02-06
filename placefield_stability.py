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
from plot_helper import ScrollPlot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos, plot_tmap_us2, plot_tmap_sm2, plot_events_over_pos2


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

    # good_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)

    return neuron_map


def fix_neuronmap(map_import):
    """
    Fixes neuronmap input imported from matlab in cell format to spit out an array and converts
    to python nomenclature (e.g. first entry = 0)
    :param map_import: poorly formatted map imported from Nat's matlab output
    :return: map_fixed: a fixed map
    """

    map_fixed = np.ones(map_import.shape, dtype=np.int64)*np.nan  #pre-allocate to NaNs
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
    good_map_bool = np.isnan(neuron_map) == 0

    # Get new neurons
    nneurons2 = ct.get_num_neurons(reg_session['Animal'], reg_session['Date'],
                                   reg_session['Session'])

    new_ind = np.where(np.isin(np.arange(0,nneurons2), neuron_map) == False)[0]

    return good_map_bool, silent_ind, new_ind


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
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)

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
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
    good_map = neuron_map[good_map_bool].astype(np.int64)

    # load in placefield objects between sessions
    PF1 = pf.load_pf(mouse, arena1, day1, pf_file='placefields_cm1_manlims.pkl')
    PF2 = pf.load_pf(mouse, arena2, day2, pf_file='placefields_cm1_manlims.pkl')

    # Identify neurons with proper mapping between sessions
    good_map_ind, _ = np.where(good_map_bool)
    ngood = len(good_map_ind)

    corrs_us = np.ndarray(ngood)  # Initialize correlation arrays
    corrs_sm = np.ndarray(ngood)
    # Step through each mapped neuron and get corrs between each
    for idn, neuron in enumerate(good_map_ind):
        reg_neuron = good_map[idn]
        corr_us, p_us = sstats.spearmanr(np.reshape(PF1.tmap_us[neuron], -1),
                                         np.reshape(PF2.tmap_us[reg_neuron], -1),
                                         nan_policy='omit')
        corr_sm, p_sm = sstats.spearmanr(np.reshape(PF1.tmap_sm[neuron], -1),
                                         np.reshape(PF2.tmap_sm[reg_neuron], -1),
                                         nan_policy='omit')
        corrs_us[idn] = corr_us
        corrs_sm[idn] = corr_sm

    return corrs_us, corrs_sm


class PFCombineObject:
    def __init__(self, mouse, arena1, day1, arena2, day2):
        self.mouse = mouse
        self.arena1 = arena1
        self.day1 = day1
        self.arena2 = arena2
        self.day2 = day2

        # load in place-field object information
        self.PF1 = pf.load_pf(mouse, arena1, day1, pf_file='placefields_cm1_manlims.pkl')
        self.PF2 = pf.load_pf(mouse, arena2, day2, pf_file='placefields_cm1_manlims.pkl')

        # Get mapping between arenas
        neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2)
        reg_session = sd.find_eraser_session(mouse, arena2, day2)
        good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
        good_map = neuron_map[good_map_bool].astype(np.int64)
        good_map_ind, _ = np.where(good_map_bool)
        self.nneurons = len(good_map_ind)

        # For loop to dump all PFs into matching lists for easy later scrolling!
        self.tmap1_us_reg = []
        self.tmap2_us_reg = []
        self.tmap1_sm_reg = []
        self.tmap2_sm_reg = []
        for idn, neuron in enumerate(good_map_ind):
            neuron_reg = good_map[idn]
            self.tmap1_us_reg.append(self.PF1.tmap_us[neuron])
            self.tmap2_us_reg.append(self.PF2.tmap_us[neuron_reg])
            self.tmap1_sm_reg.append(self.PF1.tmap_sm[neuron])
            self.tmap2_sm_reg.append(self.PF2.tmap_sm[neuron_reg])

        # Get PSAbool for co-active neurons
        self.PSAalign1 = self.PF1.PSAbool_align[good_map_ind, :]
        self.PSAalign2 = self.PF2.PSAbool_align[good_map, :]

    def pfscroll(self, current_position=0):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another

        :param current_position:
        :return:
        """

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) for n in range(self.nneurons)]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to make it through
        self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm,
                             plot_events_over_pos2, plot_tmap_us2, plot_tmap_sm2),
                            current_position=current_position, n_frames=self.nneurons,
                            n_rows=2, n_cols=3, figsize=(17.2, 10.6), titles=titles,
                            x=self.PF1.pos_align[0, self.PF1.isrunning],
                            y=self.PF1.pos_align[1, self.PF1.isrunning],
                            PSAbool=self.PSAalign1[:, self.PF1.isrunning], tmap_us=self.tmap1_us_reg,
                            tmap_sm=self.tmap1_sm_reg, x2=self.PF2.pos_align[0, self.PF2.isrunning],
                            y2=self.PF2.pos_align[1, self.PF2.isrunning],
                            PSAbool2=self.PSAalign2[:, self.PF2.isrunning],
                            tmap_us2=self.tmap2_us_reg, tmap_sm2=self.tmap2_sm_reg,
                            arena=self.PF1.arena, day=self.PF1.day, arena2=self.PF2.arena,
                            day2=self.PF2.day, mouse=self.PF1.mouse)


if __name__ == '__main__':

    # neuron_map = get_neuronmap('Marble11', 'Shock', -2, 'Shock', -1)
    # classify_cells(neuron_map)
    oratio1, oratio2, oratioboth = get_overlap('Marble11', 'Shock', -2, 'Shock', -1)

    # NRK - need to compare the below to MATLAB!!! Also need to plot sessions side-by-side
    # corrs_us, corrs_sm = pf_corr_bw_sesh('Marble11', 'Shock', -2, 'Shock', -1)
    # t = np.mean(corrs_us)

    PFcomb = PFCombineObject('Marble11', 'Shock', -2, 'Shock', -1)
    PFcomb.pfscroll()

    pass