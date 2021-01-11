# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 09:53:00 2019

@author: Nat Kinsky
"""
import importlib
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
from scipy.signal import decimate
import Placefields as pf
from mouse_sessions import make_session_list
import cell_tracking as ct
from plot_helper import ScrollPlot, pretty_plot
from er_gen_functions import plot_tmap_us, plot_tmap_sm, plot_events_over_pos, plot_tmap_us2, plot_tmap_sm2, plot_events_over_pos2
import eraser_reference as err
import skimage as ski
from skimage.transform import resize as sk_resize
from pickle import load, dump
from tqdm import tqdm
import seaborn as sns
import er_plot_functions as erp
from helpers import match_ax_lims, get_CI, mean_CI

# Plotting settings
palette = sns.color_palette('Set2')
linetypes = ['-', '--', '-.']


# Make text save as whole words
plt.rcParams['pdf.fonttype'] = 42


def get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=False):
    """
    Get mapping between registered neurons from arena1/day1 to arena2/day2
    :param mouse:
    :param arena1: session 1 day/arena
    :param day1:
    :param arena2: session 2 day/arena
    :param day2:
    :param batch_map_use: False (default) = do direct pairwise registration, True = use batch map to generate pairwise map
    (assumes batch_map lives in the Open day -2 working folder).
    :return: neuron_map: an array the length of the number of neurons in session1. NaNs indicate
    that neuron has no matched counterpart in session2. numbers indicate index of neuron in session2
    that matches session1 neuron.
    """

    make_session_list()  # Initialize session list

    if not batch_map_use:
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
        neuron_map = fix_neuronmap(map_import).reshape(-1)

        # good_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
    elif batch_map_use:
        neuron_map = get_pairwise_map_from_batch(mouse, arena1, day1, arena2, day2)

    return neuron_map


def fix_batchmap(batch_path):
    """Takes batchmap output and makes it into a nice reading format. Input = full path to batch_session_map_py.mat"""
    batch_map_import = sio.loadmat(batch_path)  # load in file in poorly formatted fashion
    map = batch_map_import['map']
    animals = [a[0][0] for a in batch_map_import['session'][0]]
    dates = [a[1][0] for a in batch_map_import['session'][0]]
    sessions = [a[2][0][0] for a in batch_map_import['session'][0]]
    session_list = [animals, dates, sessions]

    return map, session_list


def get_pairwise_map_from_batch(mouse, arena1, day1, arena2, day2):
    """
    Extracts a direct map from session1 to session2
    :param batch_map:
    :param session:
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :return:
    """
    # first load in and make batch map variables nice
    batch_path = path.join(get_dir(mouse, 'Open', -2), 'batch_session_map_py.mat')
    map, session_list = fix_batchmap(batch_path)

    # first figure out which recording session corresponds to which column in batch_map
    base_idx, reg_idx = [get_batchmap_index(session_list, mouse, arena, day) for arena, day in
                         zip([arena1, arena2], [day1, day2])]

    # now map neurons!
    map1 = np.asarray(map[:, base_idx], dtype=int) - 1  # convert from matlab to python indexing
    valid_bool = np.bitwise_not(np.isnan(map[:, base_idx]))  # id non-neg values
    neurons1 = np.arange(0, np.nanmax(map1[valid_bool] + 1))
    map2 = map[:, reg_idx] - 1  # convert from matlab to python indexing
    map2[map2 == -1] = np.nan  # Set un-mapped values to nan to match pairwise map conventions
    map1_2 = np.ones_like(neurons1, dtype=float)*np.nan
    for neuron1 in neurons1:
        id1_batch = np.where(neuron1 == map1)
        if neuron1 >= 0:
            try:
                map1_2[neuron1] = map2[id1_batch]
            except ValueError:
                t = 1

    return map1_2


def get_batchmap_index(session_list, mouse, arena, day):
    """helper function to get column # for a given session in batch_session_map"""

    session_info = sd.find_eraser_session(mouse, arena, day)
    mouse_bool = [animal == mouse for animal in session_list[0]]
    date_bool = [date == sd.fix_slash_date(session_info['Date']) for date in session_list[1]]
    session_bool = [str(session) == session_info['Session'] for session in session_list[2]]

    if np.all(mouse_bool):
        idx = np.where(np.bitwise_and(date_bool, session_bool))[0][0]
    else:
        idx = np.nan
    # add one to account for first column being a list of ALL neurons
    idx += 1
    return idx


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
    :param reg_session: index to 2nd session in session_list. Get it using code in session_directory (e.g. find_eraser_session).
    :param overlap_thresh: not functional yet. default (eventually) = consider new/silent if ROIs overlap less than
                            overlap_thresh
    :return: good_map_bool: boolean of the size of neuron_map for neurons validly mapped between sessions (e.g. not silent or new)
                silent_ind: indices of session 1 neurons that are not active/do not have a valid map in session 2.
                new_ind: indices of session 2 neurons that do not appear in session 1.
    """

    # Get silent neurons
    silent_ind = np.where(np.isnan(neuron_map))[0]
    good_map_bool = np.isnan(neuron_map) == 0

    # Get new neurons
    try:
        nneurons2 = ct.get_num_neurons(reg_session['Animal'], reg_session['Date'],
                                       reg_session['Session'])
        new_ind = np.where(np.invert(np.isin(np.arange(0, nneurons2), neuron_map)))[0]

    except FileNotFoundError:
        print('No neural data for ' + reg_session['Animal'] + ': ' + reg_session['Date'] +
              '-s' + str(reg_session['Session']))
        good_map_bool, silent_ind, new_ind = np.ones((3,))*np.nan

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


def PV1_corr_bw_sesh(mouse, arena1, day1, arena2, day2, speed_thresh=1.5,
                    pf_file='placefields_cm1_manlims_1000shuf.pkl', rot_deg=0, shuf_map=False):
    """
    Gets 1-d population vector correlations between sessions.
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param speed_thresh: speed threshold to use (default = 1.5cm/s). Not
    :param pf_file: string. Defauls = 'placefields_cm1_manlims_1000shuf.pkl'
    :param rot_deg: indicates how much to rotate day2 tmaps before calculating corrs. 0=default, 0/90/180/270 = options
    :param shuf_map: randomly shuffle neuron_map to get shuffled correlations
    :return: PVcorr_all, PVcorr_both: spearman correlation between PVs. Includes ALL neurons active in either session or
    only neurons active in both sessions
    """

    # Get mapping between sessions
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)

    # Gets PVs
    PV1 = pf.get_PV1(mouse, arena1, day1, speed_thresh=speed_thresh, pf_file=pf_file)
    PV2 = pf.get_PV1(mouse, arena2, day2, speed_thresh=speed_thresh, pf_file=pf_file)

    # Now register between sessions
    PV1all, PV2all, PV1both, PV2both = registerPV(PV1, PV2, neuron_map, reg_session, shuf_map=shuf_map)

    # Now get ALL corrs and BOTH corrs
    PVcorr_all, all_p = sstats.spearmanr(PV1all, PV2all, nan_policy='omit')
    PVcorr_both, both_p = sstats.spearmanr(PV1both, PV2both, nan_policy='omit')

    return PVcorr_all, PVcorr_both


def registerPV(PV1, PV2, neuron_map, reg_session, shuf_map=False):
    """

    :param PV1: 1-d population vector of event rates for all neurons in session 1
    :param PV2: same as above for session 2
    :param neuron_map: map between sessions obtained from SharpWave/Tenaspis github code
    :param reg_session: 2nd session id in session_list
    :return: PV1all, PV2allreg: PVs for ALL neurons active in EITHER session registered to each other -
             includes silent/new cells
             PV1both, PV2bothreg: PVs for neurons active in BOTH sessions registered to each other -
             no silent/new cells
    """

    # Identify neurons with good map
    good_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
    good_map = neuron_map[good_map_bool].astype(np.int64)

    # Shuffle neuron_map if specified
    if shuf_map:
        good_map = np.random.permutation(good_map)

    # Identify neurons with proper mapping between sessions
    try:
        good_map_ind = np.where(good_map_bool)[0]
    except ValueError:
        print('ValueError?')
    ngood = len(good_map_ind)
    nnew = len(new_ind)

    # Construct PVs with ALL neurons detected in EITHER session
    PV1all = np.concatenate((PV1, np.zeros(nnew)))
    PV2allreg = np.concatenate((np.zeros(PV1.shape), np.zeros(nnew)))
    PV2allreg[good_map_ind] = PV2[good_map]
    PV2allreg[len(PV1):] = PV2[new_ind]  # These needs checking!!!

    # Construct PVs with ONLY neurons detected in BOTH sessions
    PV1both = PV1[good_map_ind]
    PV2bothreg = PV2[good_map]

    return PV1all, PV2allreg, PV1both, PV2bothreg


def get_all_PV1corrs(mouse, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0):
    """
    Gets PV1 corrs for all sessions occurring between arena1 and arena2 for a given mouse.
    :param mouse:
    :param arena1:
    :param arena2:
    :param days: default = [-2, -1, 0, 4, 1, 2, 7]
    :param nshuf: # shuffles (default = 0).
    :return: corrs_all, corrs_both: 7x7 np-array with PV corrs between all possible session-pairs.
             shuf_all, shuf_both: 7x7xnshuf np-array with shuffled correlations.
    """

    # Pre-allocate
    corrs_both = np.ones((7, 7)) * np.nan
    corrs_all = np.ones((7, 7)) * np.nan
    shuf_both = np.ones((7, 7, nshuf)) * np.nan
    shuf_all = np.ones((7, 7, nshuf)) * np.nan
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):

            # Don't run any backward registrations
            if (arena1 == arena2 and id2 > id1) or (arena1 != arena2 and id2 >= id1):
                try:
                    PVall, PVboth = PV1_corr_bw_sesh(mouse, arena1, day1, arena2, day2)
                    corrs_both[id1, id2] = PVboth
                    corrs_all[id1, id2] = PVall
                    shuf_all[id1, id2], shuf_both[id1, id2] = PV1_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf)
                except FileNotFoundError:
                    print('File Not Found for ' + mouse + ': ' + arena1 + str(day1) + ' to ' + arena2 + str(day2))

    return corrs_all, corrs_both, shuf_all, shuf_both


def PV1_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf):
    """
    Gets correlations for 1-d PVs between arenas/days with neuron mapping shuffled between sessions.
    :param mouse:
    :param arena1: 'Shock' or 'Open'
    :param day1: [-2, -1, 0, 4, 1, 2, 7]
    :param arena2:
    :param day2:
    :param nshuf: int
    :return: shuf_all, shuf_both: (nshuf,) nd-arrays with shuffled corrs using ALL neurons or only those recorded in
    BOTH sessions
    """

    # pre-allocate lists
    temp_all = []
    temp_both = []

    # Put in something here to check for saved data if this takes too long!
    save_name = 'PV1shuf_corrs_' + arena2 + str(day2) + '_nshuf_' + str(nshuf) + '.pkl'
    dir_use = get_dir(mouse, arena1, day1)
    save_file = path.join(dir_use, save_name)

    if not path.exists(save_file):
        print('Getting shuffled 1-d PV corrs')
        for n in tqdm(np.arange(nshuf)):
            corr_all, corr_both = PV1_corr_bw_sesh(mouse, arena1, day1, arena2, day2, shuf_map=True)
            temp_all.append(corr_all)
            temp_both.append(corr_both)

        shuf_all = np.asarray(temp_all)
        shuf_both = np.asarray(temp_both)

        # pickle data for later use
        dump([['mouse', 'arena1', 'day1', 'arena2', 'day2', 'shuf_all', 'shuf_both', 'nshuf'],
              [mouse, arena1, day1, arena2, day2, shuf_all, shuf_both, nshuf]],
             open(save_file, 'wb'))
    elif path.exists(save_file):  # load previously pickled data
        print('Loading previously saved shuffled files')
        names, save_data = load(open(save_file, 'rb'))  # load it

        # Check that data matches inputs
        if save_data[0] == mouse and save_data[1] == arena1 and save_data[2] == day1 and save_data[3] == arena2 and \
                save_data[4] == day2:
            shuf_all = save_data[5]
            shuf_both = save_data[6]
        else:
            print('Previously saved data does not match inputs')
            shuf_all = np.nan
            shuf_both = np.nan

    return shuf_all, shuf_both


def pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file='placefields_cm1_manlims_1000shuf.pkl',
                    rot_deg=0, shuf_map=False, debug=False, batch_map_use=False):
    """
    Gets placefield correlations between sessions. Note that
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param pf_file: string. Defaults = 'placefields_cm1_manlims_1000shuf.pkl'
    :param rot_deg: indicates how much to rotate day2 tmaps before calculating corrs. 0=default, 0/90/180/270 = options
    :param shuf_map: randomly shuffle neuron_map to get shuffled correlations
    :return: corrs_us, corrs_sm: spearman rho values between all cells that are active
    in both sessions. us = unsmoothed, sm = smoothed
    """

    # load in placefield objects between sessions
    PF1 = pf.load_pf(mouse, arena1, day1, pf_file=pf_file)
    PF2 = pf.load_pf(mouse, arena2, day2, pf_file=pf_file)

    # Get mapping between sessions
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=batch_map_use)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)

    # only include neurons validly mapped to other neurons
    valid_map_bool, silent_ind, new_ind = classify_cells(neuron_map, reg_session)
    valid_neurons_reg = neuron_map[valid_map_bool].astype(np.int64)
    valid_neurons_base = np.where(valid_map_bool)[0]

    # Identify mapped neurons with least one calcium event after speed thresholding
    run_events_bool = np.bitwise_and(PF1.PSAboolrun[valid_neurons_base, :].sum(axis=1) > 0,
                                     PF2.PSAboolrun[valid_neurons_reg, :].sum(axis=1) > 0)

    # Refine map again to only include active neurons after speed thresholding
    good_neurons_base = valid_neurons_base[run_events_bool]
    good_neurons_reg = valid_neurons_reg[run_events_bool].astype(np.int64)

    # Shuffle mapping between sessions if specified
    if shuf_map:
        good_neurons_reg = np.random.permutation(good_neurons_reg)
    ngood = len(good_neurons_base)

    corrs_us, corrs_sm = [], []  # Initialize correlation lists

    # Step through each mapped neuron and get corrs between each
    rot = int(rot_deg/90)
    no_run_events_bool = np.ones(ngood, dtype=bool)

    for base_neuron, reg_neuron in zip(good_neurons_base, good_neurons_reg):

        # if debug and base_neuron == 364:  # for debugging nans in sstats.spearmanr
        try:
            if rot == 0 and arena1 == arena2:  # Do correlations directly if possible
                corr_us, p_us, poor_overlap_us = spearmanr_nan(PF1.tmap_us[base_neuron].reshape(-1), PF2.tmap_us[reg_neuron].reshape(-1))

                corr_sm, p_sm, poor_overlap_sm = spearmanr_nan(PF1.tmap_sm[base_neuron].reshape(-1), PF2.tmap_sm[reg_neuron].reshape(-1))

            else:  # rotate and resize PF2 before doing corrs if rotations are specified
                PF1_size = PF1.tmap_us[0].shape
                corr_us, p_us, poor_overlap_us = spearmanr_nan(PF1.tmap_us[base_neuron].reshape(-1), np.reshape(sk_resize(np.rot90(
                    PF2.tmap_us[reg_neuron], rot), PF1_size, anti_aliasing=True), -1))

                corr_sm, p_sm, poor_overlap_sm = spearmanr_nan(np.reshape(PF1.tmap_sm[base_neuron], -1), np.reshape(sk_resize(np.rot90(
                    PF2.tmap_sm[reg_neuron], rot), PF1_size, anti_aliasing=True), -1))
        except RuntimeWarning:  # Note you will have to enable warnings for this to work a la >> import warnings, >>warnings.filterwarnings('error', category=RuntimeWarning)
            print('RunTimeWarning Encountered in some basic scipy/numpy functions - should probably debug WHY this is happening')
            print('Base_neuron = ' + str(base_neuron))

        # exclude any correlations that would throw a scipy.stats.spearmanr RuntimeWarning due to
        # # poor overlap after rotation...
        if not poor_overlap_us: corrs_us.append(corr_us)
        if not poor_overlap_sm: corrs_sm.append(corr_sm)

    corrs_us, corrs_sm = np.asarray(corrs_us), np.asarray(corrs_sm)

    return corrs_us, corrs_sm


def compare_pf_at_bestrot(mouse, arena1='Shock', arena2='Shock', days=[-2, -1, 0, 4, 1, 2, 7],
                          pf_file='placefields_cm1_manlims_1000shuf.pkl', smooth=True):
    """
    Plot histogram of place-field correlations at no rotation and best rotation for all days listed.
    :param mouse:
    :param arena1:
    :param arena2:
    :param days:
    :param pf_file:
    :param smooth: False = un-smoothed correlations, True (default) = use smoothed correlations
    :return: fig_not, ax_norot, fig_bestrot, ax_bestrot: figures and axes for correlations histograms at no rotation
    and at best rotation between sessions.
    """
    if smooth:
        idcorrs = 1
        smooth_text = 'Smoothed TMaps'
    elif not smooth:
        smooth_text = 'Unsmoothed TMaps'
        idcorrs = 0
    # Load in to get cmperbin - assume same name has same value across all sessions!
    PFobj = pf.load_pf(mouse, arena=arena1, day=days[0], pf_file=pf_file)
    # Get pre-computed shuffled values for all day comparisons
    shufCIs = get_all_CIshuf(mouse, arena1, arena2, days=days, nshuf=100, pct=95)
    ndays = len(days)
    fig_norot, ax_norot = plt.subplots(ndays - 1, ndays - 1)
    fig_norot.set_size_inches([19.2, 9.28])
    fig_bestrot, ax_bestrot = plt.subplots(ndays - 1, ndays - 1)
    fig_bestrot.set_size_inches([19.2, 9.28])
    nbins = 30
    # plot each days correlations versus the other days!

    # label stuff in un-used axes for clarity
    ax_norot[1, 0].text(0.3, 0.5, 'No rotation')
    ax_bestrot[1, 0].text(0.3, 0.5, 'At best rotation')
    [a[2, 1].text(0.3, 0.5, 'cmperbin = ' + str(PFobj.cmperbin)) for a in [ax_norot, ax_bestrot]]
    [a[2, 1].text(0.3, 0.3, 'smooth =  ' + str(smooth)) for a in [ax_norot, ax_bestrot]]
    [a[2, 0].text(0.3, 0.5, arena1 + ' vs ' + arena2) for a in [ax_norot, ax_bestrot]]
    ax_bestrot[2, 0].text(0.3, 0.3, 'Red dashed = no rot. mean corr')

    # turn off axes in un-used axes for clarity - there has to be a better way to do this!
    for id1off, day1_off in enumerate(days[1:]):
        for ax_use in [ax_norot, ax_bestrot]:
            [a.axis('off') for a in ax_use[(1+id1off):, id1off]]

    # Not plot everything!
    for id1, day1 in enumerate(days[0:-1]):
        for id2, day2 in enumerate(days[(id1+1):]):
            try:
                _, best_rot, _ = get_best_rot(mouse, arena1, day1, arena2, day2,
                             pf_file='placefields_cm1_manlims_1000shuf.pkl')
                corrs_norot = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file=pf_file, shuf_map=False)
                corrs_bestrot = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file=pf_file, shuf_map=False,
                                                rot_deg=best_rot[idcorrs])
                corrs_comb = [corrs_norot, corrs_bestrot]
                for ida, ax in enumerate([ax_norot, ax_bestrot]):
                    ax[id1, id1 + id2].hist(corrs_comb[ida][idcorrs], range=(-1, 1), bins=nbins)
                    ylims = ax[id1, id1 + id2].get_ylim()
                    if ida == 1:  # plot no-rot corr mean on best rot plots
                        ax[id1, id1 + id2].plot(np.ones(2,)*np.nanmean(corrs_norot[idcorrs]), ylims, 'r-.')
                    ax[id1, id1 + id2].plot(np.ones(2,)*np.nanmean(corrs_comb[ida][idcorrs]), ylims, 'r-')
                    [ax[id1, id1 + id2].plot([a, a], ylims, style) for a, style in
                     zip(shufCIs[:, id1, id1 + id2 + 1], ['k--', 'k-', 'k--'])]
                    ax[id1, id1 + id2].set_ylim(ylims)
                    ax[id1, id1 + id2].set_title('Day ' + str(day1) + ' vs Day ' + str(day2))

            except (FileNotFoundError, TypeError):
                print('Error for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) + ' v ' + arena2 + ' Day ' + str(day2))

    return [fig_norot, fig_bestrot], [ax_norot, ax_bestrot]


def pf_corr_mean(mouse, arena1='Shock', arena2='Shock', days=[-2, -1, 0, 4, 1, 2, 7], batch_map_use=False,
                 pf_file='placefields_cm1_manlims_1000shuf.pkl', best_rot=False, nshuf=0):
    """
    Get mean placefield correlations between all sessions. Note that arena1 and arena2 must have the same size occupancy
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
    if nshuf <= 1:
        corr_mean_us, corr_mean_sm = np.ones((ndays, ndays)) * np.nan, np.ones((ndays, ndays)) * np.nan
    else:
        corr_mean_us, corr_mean_sm = np.ones((ndays, ndays, nshuf)) * np.nan, np.ones((ndays, ndays, nshuf)) * np.nan

    # loop through each pair of sessions and get the mean correlation for each session
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            # Don't loop through things you don't have reg files for
            if arena1 == arena2 and id1 < id2 or arena1 == 'Open' and arena2 == 'Shock' and id1 <= id2:
                try:
                    if best_rot:
                        corrs_temp, _, _ = get_best_rot(mouse, arena1=arena1, day1=day1, arena2=arena2, day2=day2,
                                                        pf_file=pf_file, batch_map_use=batch_map_use)
                        corr_mean_us[id1, id2] = corrs_temp[0]
                        corr_mean_sm[id1, id2] = corrs_temp[1]
                    elif not best_rot:
                        if nshuf == 0:
                            _, _, temp = get_best_rot(mouse, arena1=arena1, day1=day1, arena2=arena2, day2=day2,
                                                      pf_file=pf_file, batch_map_use=batch_map_use)
                            corr_mean_us[id1, id2] = temp[0, 0]
                            corr_mean_sm[id1, id2] = temp[1, 0]
                        elif nshuf == 1:
                            corrs_us, corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file=pf_file,
                                                                 shuf_map=True, batch_map_use=batch_map_use)
                            corr_mean_us[id1, id2] = corrs_us.mean(axis=0)
                            corr_mean_sm[id1, id2] = corrs_sm.mean(axis=0)
                            if np.isnan(corrs_us.mean(
                                    axis=0)):  # This should be obsolete now (fixed in spearman_nan below) - keep just in case!
                                print('NaN corrs in ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' vs. '
                                      + arena2 + ' day ' + str(day2) + ': INVESTIGATE!!!')
                                print('Running with np.nan for now!')
                                corr_mean_us[id1, id2] = np.nanmean(corrs_us, axis=0)
                                corr_mean_sm[id1, id2] = np.nanmean(corrs_sm, axis=0)
                        elif nshuf > 1:
                            try:
                                corr_mean_us[id1, id2, :], corr_mean_sm[id1, id2, :] = \
                                    load_shuffled_corrs(mouse, arena1, day1, arena2, day2, nshuf=nshuf)
                            except FileNotFoundError:
                                print('Shuffled data not found or Error in input.')

                except FileNotFoundError:
                    print('Missing pf files for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) +
                          ' to ' + arena2 + ' Day ' + str(day2))
                except TypeError:  # Do nothing if registering session to itself
                    print('No reg file for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) +
                          ' to ' + arena2 + ' Day ' + str(day2))

    return corr_mean_us, corr_mean_sm


def get_all_CIshuf(mouse, arena1='Shock', arena2='Shock', days=[-2, -1, 0, 4, 1, 2, 7], nshuf=1000, pct=95):
    """
    Retrieve previously calculated CIs at pct specified (95% = default) and median for all days
    :param mouse:
    :param arena1:
    :param arena2:
    :param days:
    :param nshuf: 1000 = default
    :param pct:
    :return:
    """

    # pre-allocate arrays
    ndays = len(days)
    shuf_CI = np.ones((3, ndays, ndays)) * np.nan

    # Calculate quantiles
    qtop = 1 - (100 - pct)/2/100
    qbot = (100 - pct)/2/100
    # loop through each pair of sessions and get the mean correlation for each session
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            # Don't loop through things you don't have reg files for
            if arena1 == arena2 and id1 < id2 or arena1 == 'Open' and arena2 == 'Shock' and id1 <= id2:
                try:
                    _, shuf_corrs = load_shuffled_corrs(mouse, arena1, day1, arena2, day2, nshuf)
                    shuf_CI[:, id1, id2] = np.quantile(shuf_corrs, [qbot, 0.5, qtop])
                except (FileNotFoundError, TypeError):
                    print('Missing shuffled correlation files for ' + mouse + ' ' + arena1 + ' Day ' + str(day1) +
                          ' to ' + arena2 + ' Day ' + str(day2))

    return shuf_CI


def get_best_rot(mouse, arena1='Shock', day1=-2, arena2='Shock', day2=-1, pf_file='placefields_cm1_manlims_1000shuf.pkl',
                 batch_map_use=False):
    """
    Gets the rotation of the arena in day2 that produces the best correlation. Will load previous runs from file saved in
    the appropriate directory by default
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param pf_file:
    :return: best_corr_mean: best mean correlation (un-smoothed, smoothed)
             best_rot: rotation that produces the best mean correlation (un-smoothed, smoothed)
             corr_mean_all: mean correlations at 0, 90, 180, and 270 rotation (row 1 = un-smoothed, row 2 = smoothed)
    """

    # Construct unique file save name
    save_name = 'best_rot_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + str(day2) + '_batch_map=' + \
                str(batch_map_use) + '.pkl'
    dir_use = get_dir(mouse, arena1, day1)
    save_file = path.join(dir_use, save_name)

    # All the meaningful code is here - above and below just is for saving 1st run/loading previous runs
    if not path.exists(save_file):  # only run if not already saved
        rots = [0, 90, 180, 270]
        corr_mean_all = np.empty((2, 4))
        for idr, rot in enumerate(rots):
            print(str(rot))
            try:
                corrs_us, corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                                               pf_file=pf_file, rot_deg=rot, shuf_map=False, batch_map_use=batch_map_use)
            except IndexError:  # Fix for missing sessions
                print('Index Error for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                corrs_us = corrs_sm = np.ones(1)*np.nan
            corr_mean_all[0, idr] = corrs_us.mean(axis=0)
            corr_mean_all[1, idr] = corrs_sm.mean(axis=0)

        best_rot = np.array(rots)[corr_mean_all.argmax(axis=1)]
        best_corr_mean = corr_mean_all.max(axis=1)

        # Pickle results
        dump([['Mouse', '[Arena1, Arena2]', 'day1', 'day2', 'best_corr_mean[un-smoothed, smoothed]',
               'best_rot[un-smoothed, smoothed]',
               'corr_mean_all[un-smoothed, smoothed]'],
              [mouse, [arena1, arena2], day1, day2, best_corr_mean, best_rot, corr_mean_all]],
             open(save_file, "wb"))
    elif path.exists(save_file):  # Load previous run and let user know
        print('Loading previous analysis for ' + mouse + ' ' + arena1 + ' day ' + str(day1) +
              ' to ' + arena2 + ' day ' + str(day2))
        temp = load(open(save_file, 'rb'))

        # Check to make sure you have the right data
        mousecheck = temp[1][0]
        if type(temp[1][1]) is list:  # Temp fix - get rid of once you save all the data with both arenas
            a1check = temp[1][1][0]
            a2check = temp[1][1][1]
        else:
            a1check = temp[1][1]
            a2check = temp[1][1]
        d1check = temp[1][2]
        d2check = temp[1][3]
        if mousecheck == mouse and a1check == arena1 and a2check == arena2 and d1check == day1 and d2check == day2:
            best_corr_mean = temp[1][4]
            best_rot = temp[1][5]
            corr_mean_all = temp[1][6]
        else:
            raise Exception('Loaded data does not match input data - erase and rerun')

    return best_corr_mean, best_rot, corr_mean_all


def plot_pfcorr_bygroup(corr_mean_mat, arena1, arena2, group_type, save_fig=True,
                        color='b', ax_use=None, offset=0, group_desig=1, best_rot=False,
                        prefix='PFcorrs', linetype='-'):
    """
    Scatterplot of correlations before shock, after, and several other groupings
    :param corr_mean_mat: nmice x 7 x 7 array of mean corr values for each mouse
    :param arena1: 'Shock' or 'Neutral'
    :param arena2:
    :param group_type: e.g. 'Control' or 'Anisomycin'
    :param save_fig: default = 'True' to save to eraser figures folder
    :param group_desig: 1 = include day 7 in post-shock plots, 2 = do not include day 7
    :param best_rot: True/False/Nan: add to title if 2d pf corrs performed at best rotation between arenas
    :param prefix: prefix to add to save file. e.g. '1dPVcorrs', 'PFcorrs' by default
    :return: fig, ax
    """

    nmice = corr_mean_mat.shape[0]

    # Define time epochs for scatter plots
    epochs, epoch_labels = get_time_epochs(nmice, group_desig)

    # Add in jitter to epochs
    xpts = epochs.copy() + 0.1 * np.random.standard_normal(epochs.shape)

    # Add in offset
    xpts = xpts + offset

    # Set up new fig/axes if required
    if ax_use is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_use
        fig = ax.figure

    # Plot corrs in scatterplot form
    ascat = ax.scatter(xpts.reshape(-1), corr_mean_mat.reshape(-1), color=color)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels(epoch_labels)

    ax.set_ylabel('Mean Spearman Rho')
    ax.set_title(group_type)
    unique_epochs = np.unique(epochs[~np.isnan(epochs)])
    corr_means = []
    for epoch_num in unique_epochs:
        corr_means.append(np.nanmean(corr_mean_mat[epochs == epoch_num]))
    axl = ax.plot(unique_epochs, corr_means, linetype, color=color)
    if save_fig:
        fig.savefig(path.join(err.pathname, prefix + ' ' + arena1 + ' v '
                    + arena2 + ' ' + group_type + 'group_desig' + str(group_desig) + 'best_rot' + str(best_rot) +
                    '.pdf'))

    return fig, ax, ascat, axl


def get_time_epochs(nmice, group_desig=1):
    """
    Returns groupings for plotting different time-epochs in group correlation matrices, e.g. BEFORE shock, AFTER shock,
    BEFORE v AFTER shock
    :param nmice:
    :param group_desig: 1: include day 1, 2, AND 7 in AFTER epochs, 2: include day 1 and 2 only in AFTER
    :return: epochs: nmice x 7 x 7 array with groupings for pf comparisons - see below comments for description
    """

    # Define epochs for scatter plots
    epochs = np.ones((7, 7)) * np.nan
    if group_desig == 1:

        epochs[0:2, 0:2] = 1  # 1 = before shock
        epochs[4:7, 4:7] = 2  # 2 = after shock days 1,2,7
        epochs[0:2, 4:7] = 3  # 3 = before-v-after shock
        epochs[0:2, 3] = 4  # 4 = before-v-STM
        epochs[3, 4:7] = 5  # 5 = STM-v-LTM (4hr to 1,2,7)
        epoch_labels = ['Before', 'After(Days 1-7)', 'Before v After(Days 1-7)',
                        'Before v STM', 'STM v After(Days 1-7)']
    elif group_desig == 2:
        epochs[0:2, 0:2] = 1  # 1 = before shock
        epochs[4:6, 4:6] = 2  # 2 = after shock days 1,2 only
        epochs[0:2, 4:6] = 3  # 3 = before-v-after shock
        epochs[0:2, 3] = 4  # 4 = before-v-STM
        epochs[3, 4:6] = 5  # 5 = STM-v-LTM (days 1 and 2 only)
        epoch_labels = ['Before', 'After (Day 1-2)', 'Before v After(Days1-2)',
                        'Before v STM', 'STM v After(Days1-2)']

    # now keep only values above diagonal, shape and repeat matrix to shape (nmice, 7, 7)
    epochs = np.triu(epochs, 1)
    epochs[epochs == 0] = np.nan
    epochs = np.moveaxis(np.repeat(epochs[:, :, np.newaxis], nmice, 2), 2, 0)

    return epochs, epoch_labels

def get_seq_time_pairs(nmice):
    """
    Returns pairings for plotting each session versus the next
    :return: pairs: nmice x 7 x 7 array with groupings for pf comparisons
    """

    # Define pairs for scatter plots
    pairs = np.ones((7, 7)) * np.nan
    pair_labels = ['-2 v -1', '-1 v 4hr', '4 hr v 1', '1 v 2', '2 v 7']
    pair_ids = [0, 1, 3, 4, 5]
    for idd in pair_ids:
        pairs[idd, idd+1] = idd

    # now shape and repeat matrix to shape (nmice, 7, 7)
    pairs = np.moveaxis(np.repeat(pairs[:, :, np.newaxis], nmice, 2), 2, 0)

    return pairs, pair_labels


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


def spearmanr_nan(a, b):
    """
    Perform sstats.spearmanr with nan_policy='omit with error catching to return nan if every pair of bins has at
    least one nan (i.e. the mouse never occupied any of the bins in the second session that it did in the first session)

    :param a:
    :param b:
    :return: corr, pval, poor_overlap (this last variable can be used to exclude any values at a later point)
    """

    # make sure at least 3 bins have valid numbers in each session - can't perform correlation with less than 3 points
    poor_overlap = False  # initialize this variable to track if you have poor overlap or not
    if np.sum(np.bitwise_or(np.isnan(a), np.isnan(b))) < len(a) - 2:

        good_bool = np.bitwise_and(np.bitwise_not(np.isnan(a)), np.bitwise_not(np.isnan(b)))  # Hack to fix
        # improper handling of all zero array with nan_policy='omit'

        # Handle RuntimeWarning in corrcoef where you don't get enough unique values to calculate a spearman correlation.
        if np.unique(a[good_bool]).shape[0] < 2 or np.unique(b[good_bool]).shape[0] < 2:
            corr, pval, poor_overlap = np.nan, np.nan, True
        else:
            corr, pval = sstats.spearmanr(a[good_bool], b[good_bool], nan_policy='omit')
    else:
        corr, pval = np.nan, np.nan

    return corr, pval, poor_overlap


def load_shuffled_corrs(mouse, arena1, day1, arena2, day2, nshuf):
    """
    Loads place maps correlations between sessions specified by inputs with neuron map between session shuffled nshuf times.
    :return: shuf_corrs_us_mean and shuf_corrs_sm_mean: un-smoothed and smoothed mean correlations for each shuffle
    """
    dir_use = get_dir(mouse, arena1, day1)
    file_name = 'shuffle_map_mean_corrs_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + \
                str(day2) + '_nshuf' + str(nshuf) + '.pkl'
    save_file = path.join(dir_use, file_name)

    if path.exists(save_file):
        ShufMaptemp = load(open(save_file,'rb'))
        shuf_corrs_us_mean = ShufMaptemp.shuf_corrs_us_mean
        shuf_corrs_sm_mean = ShufMaptemp.shuf_corrs_sm_mean
    else:
        print('Shuffled correlations not yet run')
        shuf_corrs_us_mean = np.nan()
        shuf_corrs_sm_mean = np.nan()

    return shuf_corrs_us_mean, shuf_corrs_sm_mean


def get_group_pf_corrs(mice, arena1, arena2, days, best_rot=False, pf_file='placefields_cm1_manlims_1000shuf.pkl',
                       batch_map_use=False, nshuf=0):
    """
    Assembles a nice matrix of mean 2d correlation values between place field maps on days/arenas specified.
    :param mice:
    :param arena1:
    :param arena2:
    :param days:
    :param best_rot:
    :param pf_file:
    :return:
    """

    # pre-allocate
    ndays = len(days)
    nmice = len(mice)
    corr_us_mean_all, corr_sm_mean_all = np.ones((nmice, ndays, ndays))*np.nan, np.ones((nmice, ndays, ndays))*np.nan
    shuf_us_mean_all, shuf_sm_mean_all = np.ones((nmice, ndays, ndays, nshuf)) * np.nan, np.ones(
        (nmice, ndays, ndays, nshuf)) * np.nan

    for idm, mouse in enumerate(mice):
        corr_us_mean_all[idm, :, :], corr_sm_mean_all[idm, :, :] = pf_corr_mean(mouse, arena1, arena2, days,
                                                                                best_rot=best_rot, pf_file=pf_file,
                                                                                batch_map_use=batch_map_use,
                                                                                nshuf=0)

    if nshuf > 0:
        for idm, mouse in enumerate(mice):
            try:
                shuf_us_mean_all[idm, :, :], shuf_sm_mean_all[idm, :, :] = \
                    pf_corr_mean(mouse, arena1, arena2, days, best_rot=False, pf_file=pf_file,
                                 batch_map_use=batch_map_use, nshuf=nshuf)
            except:
                print('Error in getting shuffled correlations')

    return corr_us_mean_all, corr_sm_mean_all, shuf_us_mean_all, shuf_sm_mean_all


def get_group_PV1d_corrs(mice, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0):
    """
    Assembles a nice matrix of mean correlation values between 1d PVs on days/arenas specified.
    :param mice:
    :param arena1:
    :param arena2:
    :param days:
    :param best_rot:
    :param pf_file:
    :return:
    """

    # pre-allocate
    ndays = len(days)
    nmice = len(mice)
    PV1_both_all, PV1_all_all = np.ones((nmice, ndays, ndays))*np.nan, np.ones((nmice, ndays, ndays))*np.nan
    PV1_both_shuf, PV1_all_shuf = np.ones((nmice, ndays, ndays, nshuf))*np.nan, np.ones((nmice, ndays, ndays, nshuf))*np.nan

    for idm, mouse in enumerate(mice):
        PV1_all_all[idm, :, :], PV1_both_all[idm, :, :], PV1_both_shuf[idm, :, :, :], PV1_all_shuf[idm, :, :, :] = \
            get_all_PV1corrs(mouse, arena1, arena2, days, nshuf=nshuf)

    return PV1_all_all, PV1_both_all, PV1_both_shuf, PV1_all_shuf


## Object to map and view placefields for same neuron mapped between different sessions
class PFCombineObject:
    def __init__(self, mouse, arena1, day1, arena2, day2, pf_file='placefields_cm1_manlims_1000shuf.pkl', debug=False):
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
        self.pval1_reg = [self.PF1.pval[a] for a in good_map_ind]

        # Get correlations between sessions!
        self.corrs_us, self.corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                                                       pf_file=pf_file, debug=debug)

    def pfscroll(self, current_position=0, pval_thresh=1, best_rot=False):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another.

        :param current_position: starting index of neurons in pval_thresholded neuron array. A bit clunky since
        you don't know which neurons are in the thresholded version until below.
        :param pval_thresh: default = 1. Only scroll through neurons with pval (based on mutual information scores
        calculated after circularly permuting calcium traces/events) < pval_thresh
        :return:
        """

        # Get only spatially tuned neurons: those with mutual spatial information pval < pval_thresh
        spatial_neurons = [a < pval_thresh for a in self.pval1_reg]

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) + " best_rot=" + str(best_rot) for n in np.where(spatial_neurons)[0]]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to make it through
        lims1 = [[self.PF1.xEdges.min(), self.PF1.xEdges.max()], [self.PF1.yEdges.min(), self.PF1.yEdges.max()]]
        lims2 = [[self.PF2.xEdges.min(), self.PF2.xEdges.max()], [self.PF2.yEdges.min(), self.PF2.yEdges.max()]]
        if not best_rot:
            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm,
                                 plot_events_over_pos2, plot_tmap_us2, plot_tmap_sm2),
                                current_position=current_position, n_neurons=self.nneurons,
                                n_rows=2, n_cols=3, figsize=(17.2, 10.6), titles=titles,
                                x=self.PF1.pos_align[0, self.PF1.isrunning],
                                y=self.PF1.pos_align[1, self.PF1.isrunning],
                                PSAbool=self.PSAalign1[spatial_neurons, :][:, self.PF1.isrunning],
                                tmap_us=[self.tmap1_us_reg[a] for a in np.where(spatial_neurons)[0]],
                                tmap_sm=[self.tmap1_sm_reg[a] for a in np.where(spatial_neurons)[0]],
                                x2=self.PF2.pos_align[0, self.PF2.isrunning],
                                y2=self.PF2.pos_align[1, self.PF2.isrunning],
                                PSAbool2=self.PSAalign2[spatial_neurons, :][:, self.PF2.isrunning],
                                tmap_us2=[self.tmap2_us_reg[a] for a in np.where(spatial_neurons)[0]],
                                tmap_sm2=[self.tmap2_sm_reg[a] for a in np.where(spatial_neurons)[0]],
                                arena=self.PF1.arena, day=self.PF1.day, arena2=self.PF2.arena,
                                day2=self.PF2.day, mouse=self.PF1.mouse,
                                corrs_us=self.corrs_us[spatial_neurons], corrs_sm=self.corrs_sm[spatial_neurons],
                                traj_lims=lims1, traj_lims2=lims2)
        elif best_rot:
            if best_rot == 90 or best_rot == 270:  # change limits if rotated 90 or 270 degrees
                lims2 = [[self.PF2.yEdges.min(), self.PF2.yEdges.max()],
                         [self.PF2.xEdges.min(), self.PF2.xEdges.max()]]
            _, best_rot, _ = get_best_rot(self.mouse, self.arena1, self.day1, self.arena2, self.day2)
            # best_rot is spit out for un-smoothed and smoothed place maps. use smoothed [1]
            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm,
                                 plot_events_over_pos2, plot_tmap_us2, plot_tmap_sm2),
                                current_position=current_position, n_neurons=self.nneurons,
                                n_rows=2, n_cols=3, figsize=(17.2, 10.6), titles=titles,
                                x=self.PF1.pos_align[0, self.PF1.isrunning],
                                y=self.PF1.pos_align[1, self.PF1.isrunning],
                                PSAbool=self.PSAalign1[spatial_neurons, :][:, self.PF1.isrunning],
                                tmap_us=[self.tmap1_us_reg[a] for a in np.where(spatial_neurons)[0]],
                                tmap_sm=[self.tmap1_sm_reg[a] for a in np.where(spatial_neurons)[0]],
                                x2=self.PF2.pos_align[0, self.PF2.isrunning],
                                y2=self.PF2.pos_align[1, self.PF2.isrunning],
                                PSAbool2=self.PSAalign2[spatial_neurons, :][:, self.PF2.isrunning],
                                tmap_us2=[np.rot90(self.tmap2_us_reg[a], best_rot[1]/90) for a in np.where(spatial_neurons)[0]],
                                tmap_sm2=[np.rot90(self.tmap2_sm_reg[a], best_rot[1]/90) for a in np.where(spatial_neurons)[0]],
                                arena=self.PF1.arena, day=self.PF1.day, arena2=self.PF2.arena,
                                day2=self.PF2.day, mouse=self.PF1.mouse,
                                corrs_us=self.corrs_us[spatial_neurons], corrs_sm=self.corrs_sm[spatial_neurons],
                                traj_lims=lims1, traj_lims2=lims2)


# Create class to calculate and save correlations between sessions with neuron_map shuffled
class ShufMap:
    def __init__(self, mouse, arena1='Shock', day1=-2, arena2='Shock', day2=-1, nshuf=100):
        self.mouse = mouse
        self.arena1 = arena1
        self.arena2 = arena2
        self.day1 = day1
        self.day2 = day2
        self.nshuf = nshuf
        self.shuf_corrs_us_mean = []
        self.shuf_corrs_sm_mean = []
        self.save_file = path.join(get_dir(self.mouse, self.arena1, self.day1), 'shuffle_map_mean_corrs_' + self.arena1
                                   + 'day' + str(self.day1) + '_' + self.arena2 + 'day' +
                                   str(self.day2) + '_nshuf' + str(self.nshuf) + '.pkl')

    def get_shuffled_corrs(self):  # Get tmap correlations after shuffling neuron_map
        shuf_corrs_us_mean = []
        shuf_corrs_sm_mean = []
        print('Running Shuffled Map Corrs for ' + self.mouse + ' ' + self.arena1 + ' day ' + str(self.day1) + ' to ' +
              self.arena2 + ' day ' + str(self.day2))
        for n in tqdm(np.arange(self.nshuf)):
            shuf_corrs_us, shuf_corrs_sm = pf_corr_bw_sesh(self.mouse, self.arena1, self.day1, self.arena2, self.day2,
                                                           shuf_map=True)
            shuf_corrs_us_mean.append(shuf_corrs_us.mean(axis=0))
            shuf_corrs_sm_mean.append(shuf_corrs_sm.mean(axis=0))

        self.shuf_corrs_us_mean = shuf_corrs_us_mean
        self.shuf_corrs_sm_mean = shuf_corrs_sm_mean

    def save_data(self):  # Save data to pickle file
        # dump into pickle file with name
        with open(self.save_file, 'wb') as output:
            dump(self, output)


## create a class to construct and keep all group data in a nice format and plot things...
class GroupPF:
    def __init__(self):
        self.amice = err.ani_mice_good
        self.lmice = err.learners
        self.nlmice = err.nonlearners
        self.days = [-2, -1, 0, 4, 1, 2, 7]

    def _save(self, dir=r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser'):
        dump(self.data, open(path.join(dir, 'group_data_rot=' + str(self.best_rot) + '.pkl'), 'wb'))
        return None

    def _load(self, dir=r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser', best_rot=True):
        self.data = load(open(path.join(dir, 'group_data_rot=' + str(best_rot) + '.pkl'), 'rb'))
        self.best_rot = best_rot

    def construct(self, types=['PFsm', 'PFus', 'PV1dboth', 'PV1dall'], best_rot=True,
                  pf_file='placefields_cm1_manlims_1000shuf.pkl', nshuf=1000):
        """Sets up all data in well-organized dictionary: data[type]['data' or 'shuf'][group][arena_type] where
        arena_type=0 for Open, 1 for Shock, and 2 for Open v Shock"""
        # perform PFcorrs at best rotation between session if True, False = no rotation
        groups = ['Learners', 'Nonlearners', 'Ani']
        # group_dict = dict.fromkeys(groups, {'corrs': [], 'shuf': []})
        self.data = dict.fromkeys(types)  # pre-allocate
        self.best_rot = best_rot
        self.nshuf = nshuf
        self.cmperbin = pf.load_pf(self.lmice[0], 'Shock', -2, pf_file=pf_file).cmperbin
        for type in types:
            learn_bestcorr_mean_all, nlearn_bestcorr_mean_all, ani_bestcorr_mean_all = [], [], []
            learn_shuf_all, nlearn_shuf_all, ani_shuf_all = [], [], []

            for ida, arena in enumerate(['Open', 'Shock', ['Open', 'Shock']]):
                if isinstance(arena, list):
                    arena1, arena2 = arena[0], arena[1]
                else:
                    arena1, arena2 = arena, arena
                if type == 'PFsm':
                    a, templ, b, temp_sh_l = get_group_pf_corrs(self.lmice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                    _, tempnl, _, temp_sh_nl = get_group_pf_corrs(self.nlmice, arena1, arena2, self.days,
                                                                  best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                    _, tempa, _, temp_sh_a = get_group_pf_corrs(self.amice, arena1, arena2, self.days,
                                                  best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                elif type == 'PFus':
                    templ, _, temp_sh_l, _ = get_group_pf_corrs(self.lmice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                    tempnl, _, temp_sh_nl, _ = get_group_pf_corrs(self.nlmice, arena1, arena2, self.days,
                                                                  best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                    tempa, _, temp_sh_a, _ = get_group_pf_corrs(self.amice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf)
                elif type == 'PV1dboth':
                    _, templ, _, temp_sh_l = get_group_PV1d_corrs(self.lmice, arena1, arena2, self.days, nshuf=nshuf)
                    _, tempnl, _, temp_sh_nl = get_group_PV1d_corrs(self.nlmice, arena1, arena2, self.days, nshuf=nshuf)
                    _, tempa, _, temp_sh_a = get_group_PV1d_corrs(self.amice, arena1, arena2, self.days, nshuf=nshuf)
                elif type == 'PV1dall':
                    templ, _, temp_sh_l, _ = get_group_PV1d_corrs(self.lmice, arena1, arena2, self.days, nshuf=nshuf)
                    tempnl, _, temp_sh_nl, _ = get_group_PV1d_corrs(self.nlmice, arena1, arena2, self.days, nshuf=nshuf)
                    tempa, _, temp_sh_a, _ = get_group_PV1d_corrs(self.amice, arena1, arena2, self.days, nshuf=nshuf)

                learn_bestcorr_mean_all.append(templ)
                nlearn_bestcorr_mean_all.append(tempnl)
                ani_bestcorr_mean_all.append(tempa)
                learn_shuf_all.append(temp_sh_l)
                nlearn_shuf_all.append(temp_sh_nl)
                ani_shuf_all.append(temp_sh_a)

            data_comb = [learn_bestcorr_mean_all, nlearn_bestcorr_mean_all, ani_bestcorr_mean_all]
            shuf_comb = [learn_shuf_all, nlearn_shuf_all, ani_shuf_all]

            # Now organize everything nicely into a dictionary.
            self.data[type] = {'data': [], 'shuf': []}
            self.data[type]['data'] = {group: data for group, data in zip(groups, data_comb)}
            self.data[type]['shuf'] = {group: shuf for group, shuf in zip(groups, shuf_comb)}
            self.data['best_rot'] = best_rot

    def scatter_epochs(self, arena1='Shock', arena2='Shock', groups=['Learners', 'Nonlearners', 'Ani'], type='PFsm',
                       group_desig=2, save_fig=False, ax_use=None):
        """
        Scatter plot across all time epochs
        :param arena1: 'Shock' or 'Open'
        :param arena2: 'Shock' or 'Open'
        :param groups: list of ['Learners', 'Nonlearners', 'Ani'] or any combo thereof
        :param type: 'PFsm', 'PFus', 'PV1dall', 'PV1dboth
        :param group_desig: 1 = use day 1, 2, & 7 for AFTER, 2 = use 1 & 2 only for AFTER
        :return:
        """

        # Set up plots
        fig, ax = self.figset(ax_use)
        idc = self.idmat(arena1, arena2)
        save_flag = False  # Set up saving plots

        ascat, axlines = [], []
        linetypes = ['-', '--', '-.']
        for idg, group in enumerate(groups):
            if idg == (len(groups) - 1) and save_fig is True:
                save_flag = True

            _, _, atemp, axl = plot_pfcorr_bygroup(self.data[type]['data'][group][idc], arena1, arena2,
                                                   type + ' Correlations ' + arena1 + ' v ' + arena2,
                                                   ax_use=ax, color=palette[idg], offset=-0.1, save_fig=save_flag,
                                                   group_desig=group_desig, linetype=linetypes[idg])
            ascat.append(atemp)
            axlines.append(axl)
        ax.legend(groups)

        return fig, ax, ascat, axlines

    def plot_conf(self, arena1='Shock', arena2='Shock', groups=['Learners', 'Nonlearners', 'Ani'], type='PFsm',
                       save_fig=False, ax_use=None):
        """Plot confusion matrices across groups - probably not that useful, but maybe?"""
        ndays = len(self.days)

        # Set up plots
        fig, ax = self.figset(ax_use, nplots=[1, 3], size=[16.25, 4.65])
        idc = self.idmat(arena1, arena2)
        save_flag = False  # Set up saving plots

        ascat, axlines = [], []
        for idg, (a, group) in enumerate(zip(ax, groups)):
            plot_confmat(np.nanmean(self.data[type]['data'][group][idc], axis=0), arena1, arena2, group,
                        ndays=ndays, ax_use=a)

            # Save fig after last plot if specified
            if idg == (len(groups) - 1) and save_fig is True:
                fig.savefig(self.gen_savename('Confmat', type, arena1, arena2, self.best_rot))

    def scatterbar_bw_groups(self, groups=['Learners', 'Ani', 'Nonlearners'], type='PFsm',
                             group_desig=2, save_fig=False, match_yaxis=True):
        """Scatterbar plots between groups for all epochs at once?"""

        # Set up data
        epoch_mat = []
        for group in groups:
            nmice = self.data[type]['data'][group][0].shape[0]
            etemp, epoch_labels = get_time_epochs(nmice, group_desig)
            epoch_mat.append(etemp)

        plot_title = 'No rotation'
        if self.best_rot:
            plot_title = 'Optimum rotation'


        data_use, shuf_use = self.data[type]['data'], self.data[type]['shuf']
        epoch_nums = np.unique(epoch_mat[0][~np.isnan(epoch_mat[0])]).tolist()
        axes, figs, savenames = [], [], []
        nshuf = shuf_use[groups[0]][0].shape[3]
        for ide, epoch_num in enumerate(epoch_nums):
            # Assemble actual data
            open_corrs = [data_use[group][0][epoch_mat[idg] == epoch_num] for idg, group in enumerate(groups)]
            shock_corrs = [data_use[group][1][epoch_mat[idg] == epoch_num] for idg, group in enumerate(groups)]

            # Assemble shuffled data and get mean CIs for each group
            open_CIs = [mean_CI(shuf_use[group][0].reshape(-1, nshuf)[epoch_mat[idg].reshape(-1) == epoch_num])
                         for idg, group in enumerate(groups)]
            shock_CIs = [mean_CI(shuf_use[group][1].reshape(-1, nshuf)[epoch_mat[idg].reshape(-1) == epoch_num])
                         for idg, group in enumerate(groups)]

            fig, ax, pval, tstat = erp.pfcorr_compare(open_corrs, shock_corrs, group_names=groups,
                                                      xlabel=epoch_labels[ide], ylabel=type,
                                                      xticklabels=['Open', 'Shock'],
                                                      CIs=[open_CIs, shock_CIs])

            ax[0].set_title(plot_title)  # label plot
            # Track figure/axes handles and save names
            axes.append(ax[0])
            figs.append(fig)
            savename = path.join(err.pathname, type + ' 2x2 All Groups ' + epoch_labels[ide] + '.pdf')
            savenames.append(savename)

        # Now adjust axes to all have the same min/max values.
        if match_yaxis:
            match_ax_lims(axes, type='y')

        # Save all if indicated
        if save_fig:
            [fig.savefig(savename) for fig, savename in zip(figs, savenames)]

    def figset(self, ax=None, nplots=[1, 1], size=[9.27, 3.36]):
        """Set up figure"""
        # Set up plots
        if ax is None:
            fig, ax = plt.subplots(nplots[0], nplots[1])
            fig.set_size_inches(size)
        else:
            fig = ax.figure

        return fig, ax

    def idmat(self, arena1, arena2):
        """Get appropriate index for correlations in list"""
        ascat, axlines = [], []
        if arena1 == arena2:
            if arena1 == 'Open':
                idc = 0  # Open v Open
            else:
                idc = 1  # Shock v Shock
        else:
            idc = 2  # Open v Shock

        return idc

    def gen_savename(self, prefix, type, arena1, arena2, append):
        savename = path.join(err.pathname, prefix + '_' + type + '_' + arena1 + 'v'
                             + arena2 + '_best_rot=' + str(self.best_rot) + append + '.pdf')
        return savename

# class PFrotObj:
#     def __init__(self, mouse, arena1, day1, arena2, day2):
#         self.mouse = mouse
#         self.arena1 = arena1
#         self.arena2 = arena2
#         self.day1 = day1
#         self.day2 = day2
#
#         best_corr_mean, best_rot, corr_mean_all = get_best_rot(self.mouse, self.arena1, self.day1,
#                                                                self.arena2, self.day2)


if __name__ == '__main__':
    pfg = GroupPF()
    pfg._load()
    pfg.scatterbar_bw_groups()

    pass

