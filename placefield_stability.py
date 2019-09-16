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
import eraser_reference as err
import skimage as ski
from skimage.transform import resize as sk_resize

def get_neuronmap(mouse, arena1, day1, arena2, day2):
    """
    Get mapping between registered neurons from arena1/day1 to arena2/day2
    :param mouse:
    :param arena1: session 1 day/arena
    :param day1:
    :param arena2: session 2 day/arena
    :param day2:
    :return: neuron_map: an array the length of the number of neurons in session1. NaNs indicate
    that neuron has no matched counterpart in session2. numbers indicate index of neuron in session2
    that matches session1 neuron.
    """

    make_session_list()  # Initialize session list

    # Identify map file
    dir_use = get_dir(mouse, arena1, day1)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    reg_filename = 'neuron_map-' + mouse + '-' + sd.fix_slash_date(reg_session['Date']) + \
                   '-session' + reg_session['Session'] + '.mat'
    map_file = path.join(dir_use, reg_filename)

    # Load file in
    try:
        map_data = sio.loadmat(map_file)
    except TypeError:
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
    Classifies cells as good, silent, and new based on the input mapping from session 1 to session 2 in neuron_map.
    :param neuron_map: an ndarray (#neurons in session 1 x 1) with the neuron index in session 2 that maps to the
                       index in session 1. NaN = no mapping/silent cell in session 2.
    :param overlap_thresh: not functional yet. default (eventually) = consider new/silent if ROIs overlap less than
                            overlap_thresh
    :return: good_map_bool: boolean of the size of neuron_map for neurons validly mapped between sessions
                silent_ind: indices of session 1 neurons that are not active/do not have a valid map in session 2.
                new_ind: indices of session 2 neurons that do not appear in session 1.
    """

    # Get silent neurons
    silent_ind, _ = np.where(np.isnan(neuron_map))
    good_map_bool = np.isnan(neuron_map) == 0

    # Get new neurons
    nneurons2 = ct.get_num_neurons(reg_session['Animal'], reg_session['Date'],
                                   reg_session['Session'])

    new_ind = np.where(np.invert(np.isin(np.arange(0, nneurons2), neuron_map)))[0]

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
            # overlapping cells between sessions divided by the
            the number of cells active in 1st/2nd/both sessions
            overlap_ratio_max/min: same as above but divided by max/min number
            of cells active in either session
    """
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)

    num_active1 = len(good_map_bool)
    num_active2 = sum(good_map_bool) + len(new_ind)
    num_active_min = min(num_active1, num_active2)
    num_active_max = max(num_active1, num_active2)
    num_active_both = len(good_map_bool) + len(new_ind)
    num_overlap = sum(good_map_bool)

    overlap_ratio1 = num_overlap/num_active1
    overlap_ratio2 = num_overlap/num_active2
    overlap_ratio_both = num_overlap/num_active_both
    overlap_ratio_min = num_overlap/num_active_min
    overlap_ratio_max = num_overlap/num_active_max

    return overlap_ratio1, overlap_ratio2, overlap_ratio_both, overlap_ratio_min, overlap_ratio_max


def pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                    pf_file='placefields_cm1_manlims_1000shuf.pkl', rot_deg=0, shuf_map=False):
    """
    Gets placefield correlations between sessions. Note that
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param pf_file: string. Defauls = 'placefields_cm1_manlims_1000shuf.pkl'
    :param rot_deg: indicates how much to rotate day2 tmaps before calculating corrs. 0=default, 0/90/180/270 = options
    :param shuf_map: randomly shuffle neuron_map to get shuffled correlations
    :return: corrs_us, corrs_sm: spearman rho values between all cells that are active
    in both sessions. us = unsmoothed, sm = smoothed
    """

    # Get mapping between sessions
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
    good_map = neuron_map[good_map_bool].astype(np.int64)

    # Shuffle neuron_map if specified
    if shuf_map:
        good_map = np.random.permutation(good_map)

    # load in placefield objects between sessions
    PF1 = pf.load_pf(mouse, arena1, day1, pf_file=pf_file)
    PF2 = pf.load_pf(mouse, arena2, day2, pf_file=pf_file)

    # Identify neurons with proper mapping between sessions
    good_map_ind, _ = np.where(good_map_bool)
    ngood = len(good_map_ind)

    corrs_us = np.ndarray(ngood)  # Initialize correlation arrays
    corrs_sm = np.ndarray(ngood)
    # Step through each mapped neuron and get corrs between each
    rot = int(rot_deg/90)

    if ski.__version__ < '0.16.0':
        print('WARNING - using an old version of scikit-image - update to 0.16!')
    for idn, neuron in enumerate(good_map_ind):
        reg_neuron = good_map[idn]
        if rot == 0:  # Do correlations directly if possible
            corr_us, p_us = sstats.spearmanr(np.reshape(PF1.tmap_us[neuron], -1),
                                         np.reshape(PF2.tmap_us[reg_neuron], -1),
                                         nan_policy='omit')
            corr_sm, p_sm = sstats.spearmanr(np.reshape(PF1.tmap_sm[neuron], -1),
                                         np.reshape(PF2.tmap_sm[reg_neuron], -1),
                                         nan_policy='omit')
        else:  # rotate and resize PF2 before doing corrs if rotations are specified
            PF1_size = PF1.tmap_us[0].shape
            try:  # this shouldn't be necessary once scikit-image is updated to 0.16
                corr_us, p_us = sstats.spearmanr(np.reshape(PF1.tmap_us[neuron], -1),
                                             np.reshape(sk_resize(np.rot90(PF2.tmap_us[reg_neuron], rot),
                                             PF1_size, anti_aliasing=True), -1),
                                             nan_policy='omit')
                corr_sm, p_sm = sstats.spearmanr(np.reshape(PF1.tmap_sm[neuron], -1),
                                             np.reshape(sk_resize(np.rot90(PF2.tmap_sm[reg_neuron], rot),
                                             PF1_size, anti_aliasing=True), -1),
                                             nan_policy='omit')

            except TypeError:  # display warning if you have an older scikit-image package. anti_aliasing only exists in  0.16
                corr_us, p_us = sstats.spearmanr(np.reshape(PF1.tmap_us[neuron], -1),
                                                 np.reshape(sk_resize(np.rot90(PF2.tmap_us[reg_neuron], rot),
                                                 PF1_size), -1), nan_policy='omit')
                corr_sm, p_sm = sstats.spearmanr(np.reshape(PF1.tmap_sm[neuron], -1),
                                                 np.reshape(sk_resize(np.rot90(PF2.tmap_sm[reg_neuron], rot),
                                                 PF1_size), -1), nan_policy='omit')

        corrs_us[idn] = corr_us
        corrs_sm[idn] = corr_sm

    return corrs_us, corrs_sm


def pf_corr_mean(mouse, arena1='Shock', arena2='Shock', days=[-2, -1, 0, 4, 1, 2, 7],
                 pf_file='placefields_cm1_manlims_1000shuf.pkl'):
    """
    Get mean placefield correlations between all sessions. Note that arena1 and arena2 must have the same size occpupancy
    maps already ran for each arena (tmap_us and tmap_sm arrays in Placefield object defined in Placefields.py)
    :param mouse: str of mousename
    :param arena1: str of arena1
    :param arena2: str of arena2
    :param days: list of ints of day. Valid options = [-2,-1, 0, 4, 1, 2, 7]
    :param pf_file: filename for PF file to use. default = 'placefields_cm1_manlims_1000shuf.pkl'
    :return: corr_mean_us/sm: mean spearman correlation for placefields between sessions.
    ndays x ndays ndarray. rows = arena1, columns = arena2.
    """

    # pre-allocate arrays
    ndays = len(days)
    corr_mean_us = np.ones((ndays, ndays)) * np.nan
    corr_mean_sm = np.ones((ndays, ndays)) * np.nan

    # loop through each pair of sessions and get the mean correlation for each session
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            if id1 < id2:  # Don't loop through things you don't have reg files for
                try:
                    corrs_us, corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file=pf_file)
                    corr_mean_us[id1, id2] = corrs_us.mean(axis=0)
                    corr_mean_sm[id1, id2] = corrs_sm.mean(axis=0)
                    if np.isnan(corrs_us.mean(axis=0)):
                        print('NaN corrs in ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' vs. '
                              + arena2 + ' day ' + str(day2) + ': INVESTIGATE!!!')
                        print('Running with np.nan for now!')
                        corr_mean_us[id1, id2] = np.nanmean(corrs_us, axis=0)
                        corr_mean_sm[id1, id2] = np.nanmean(corrs_sm, axis=0)
                except FileNotFoundError:
                    print('Missing pf files for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) +
                          ' to ' + arena2 + ' Day ' + str(day2))
                except TypeError:  # Do nothing if registering session to itself
                    print('No reg file for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) +
                          ' to ' + arena2 + ' Day ' + str(day2))

    return corr_mean_us, corr_mean_sm


def plot_pfcorr_bygroup(corr_mean_mat, arena1, arena2, group_type, save_fig=True,
                        color='b', ax_use=None, offset=0, group_desig=1):
    """
    Scatterplot of correlations before shock, after, and several other groupings
    :param corr_mean_mat: nmice x 7 x 7 array of mean corr values for each mouse
    :param arena1: 'Shock' or 'Neutral'
    :param arena2:
    :param group_type: e.g. 'Control' or 'Anisomycin'
    :param save_fig: default = 'True' to save to eraser figures folder
    :param group_desig: 1 = include day 7 in post-shock plots, 2 = do not include day 7
    :return: fig, ax
    """

    # Define groups for scatter plots
    if group_desig == 1:
        groups = np.ones_like(corr_mean_mat) * np.nan
        groups[:, 0:2, 0:2] = 1  # 1 = before shock
        groups[:, 4:7, 4:7] = 2  # 2 = after shock days 1,2,7
        groups[:, 0:2, 4:7] = 3  # 3 = before-v-after shock
        groups[:, 0:2, 3] = 4  # 4 = before-v-STM
        groups[:, 3, 4:7] = 5  # 5 = STM-v-LTM
    elif group_desig == 2:
        groups = np.ones_like(corr_mean_mat) * np.nan
        groups[:, 0:2, 0:2] = 1  # 1 = before shock
        groups[:, 4:6, 4:6] = 2  # 2 = after shock days 1,2 only
        groups[:, 0:2, 4:6] = 3  # 3 = before-v-after shock
        groups[:, 0:2, 3] = 4  # 4 = before-v-STM
        groups[:, 3, 4:6] = 5  # 5 = STM-v-LTM (days 1 and 2 only)

    # Add in jitter to groups
    xpts = groups.copy() + 0.1 * np.random.standard_normal(groups.shape)

    # Add in offset
    xpts = xpts + offset

    # Set up new fig/axes if required
    if ax_use is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_use
        fig = ax.figure

    # Plot corrs in scatterplot form
    ax.scatter(xpts.reshape(-1), corr_mean_mat.reshape(-1), color=color)
    ax.set_xticks(np.arange(1, 6))
    if group_desig == 1:
        ax.set_xticklabels(['Before Shk', 'After Shk (Days1-7)', 'Bef-Aft',
                            'Bef-STM', 'STM-Aft'])
    elif group_desig == 2:
        ax.set_xticklabels(['Before Shk', 'After Shk (Days1,2 only)', 'Bef-Aft',
                            'Bef-STM', 'STM-Aft'])

    ax.set_ylabel('Mean Spearman Rho')
    ax.set_title(group_type)
    unique_groups = np.unique(groups[~np.isnan(groups)])
    corr_means = []
    for group_num in unique_groups:
        corr_means.append(np.nanmean(corr_mean_mat[groups == group_num]))
    ax.plot(unique_groups, corr_means, color + '-')
    if save_fig:
        fig.savefig(path.join(err.pathname, 'PFcorrs ' + arena1 + ' v '
                    + arena2 + ' ' + group_type + 'group_desig' + str(group_desig) + '.pdf'))

    return fig, ax


def plot_confmat(corr_mean_mat, arena1, arena2, group_type, ndays=7, ax_use=None, save_fig=True):
    """
    Plots confusion matrix of mean pf corrs fo all sessions in arena1 v arena2
    :param corr_mean_mat: 7x7 array of mean pf corrs for mouse/mice
    :param arena1:
    :param arena2:
    :param group_type:
    :param ax_use:
    :param save_fig:
    :return:
    """

    # Set up fig/axes if not specified
    if ax_use is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_use
        fig = ax_use.figure

    # Plot corrs in confusion matrix
    ax.imshow(corr_mean_mat)
    ax.set_xlim((0.5, ndays - 0.5))
    ax.set_ylim((ndays - 1.5, -0.5))
    ax.set_xticklabels(['-2', '-1', '0', '4hr', '1', '2', '7'])
    ax.set_yticklabels([' ', '-2', '-1', '0', '4hr', '1', '2', '7'])
    ax.set_xlabel(arena2 + ' Day #')
    ax.set_ylabel(arena1 + ' Day #')
    ax.set_title(' Mean Spearman Rho: ' + group_type)

    if save_fig:
        fig.savefig(path.join(err.pathname, 'PFcorr Matrices ' + arena1 + ' v '
                              + arena2 + ' ' + group_type + '.pdf'))


class PFCombineObject:
    def __init__(self, mouse, arena1, day1, arena2, day2, pf_file='placefields_cm1_manlims_1000shuf.pkl'):
        self.mouse = mouse
        self.arena1 = arena1
        self.day1 = day1
        self.arena2 = arena2
        self.day2 = day2

        # load in place-field object information
        self.PF1 = pf.load_pf(mouse, arena1, day1, pf_file=pf_file)
        self.PF2 = pf.load_pf(mouse, arena2, day2, pf_file=pf_file)

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

        # Get correlations between sessions!
        self.corrs_us, self.corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                                                       pf_file=pf_file)

    def pfscroll(self, current_position=0):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another.

        :param current_position:
        :return:
        """

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) for n in range(self.nneurons)]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to make it through
        lims1 = [[self.PF1.xEdges.min(), self.PF1.xEdges.max()], [self.PF1.yEdges.min(), self.PF1.yEdges.max()]]
        lims2 = [[self.PF2.xEdges.min(), self.PF2.xEdges.max()], [self.PF2.yEdges.min(), self.PF2.yEdges.max()]]
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
                            day2=self.PF2.day, mouse=self.PF1.mouse,
                            corrs_us=self.corrs_us, corrs_sm=self.corrs_sm,
                            traj_lims=lims1, traj_lims2=lims2)


if __name__ == '__main__':
    pf_corr_bw_sesh('Marble24', 'Shock', -2, 'Shock', -1, rot_deg=90)

    pass
