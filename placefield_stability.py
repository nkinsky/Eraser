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
from matplotlib import axes
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
color_dict = {'learners': palette[0], 'nonlearners': palette[1], 'ani': palette[2]}
linetypes = ['-', '--', '-.']
linetype_dict = {'learners': linetypes[0], 'nonlearners': linetypes[1], 'ani': linetypes[2]}


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


def PV1_corr_bw_sesh(mouse, arena1, day1, arena2, day2, speed_thresh=1.5, batch_map_use=True,
                    pf_file='placefields_cm1_manlims_1000shuf.pkl', shuf_map=False):
    """
    Gets 1-d population vector correlations between sessions.
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param speed_thresh: speed threshold to use (default = 1.5cm/s). Not
    :param pf_file: string. Defauls = 'placefields_cm1_manlims_1000shuf.pkl'
    :param shuf_map: randomly shuffle neuron_map to get shuffled correlations
    :return: PVcorr_all, PVcorr_both: spearman correlation between PVs. Includes ALL neurons active in either session or
    only neurons active in both sessions
    """

    # Get mapping between sessions
    # try:
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=batch_map_use)
    # except IndexError:
    #     print('debugging PV1corr_bw_sesh')
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


def PV2_corr_bw_sesh(mouse, arena1, day1, arena2, day2, speed_thresh=1.5, corr_type='sm', batch_map_use=True,
                     pf_file='placefields_cm1_manlims_1000shuf.pkl', best_rot=False, shuf_map=False,
                     perform_corr=True):
    """
    Gets 2-d population vector correlations between sessions.
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    :param speed_thresh: speed threshold to use (default = 1.5cm/s).
    :param corr_type: 'sm' (default) for smoothed or 'us' for unsmoothed event map correlations
    :param pf_file: string. Default = 'placefields_cm1_manlims_1000shuf.pkl'
    :param best_rot: boolean, False(default) = leave maps as is, True = rotate 2nd session the amount that produces the
    best correlation.
    :param shuf_map: randomly shuffle neuron_map to get shuffled correlations
    :param perform_corr: False = don't do corr. If False, spits out PVcorr_all as (PV1all, PV2all) and PVcorr_both as
    (PV1both, PV2both) for easy re-shuffling later on. True = default.
    :return: PVcorr_all, PVcorr_both: spearman correlation between PVs. Includes ALL neurons active in either session or
    only neurons active in both sessions
    """

    # Get mapping between sessions
    neuron_map = get_neuronmap(mouse, arena1, day1, arena2, day2, batch_map_use=batch_map_use)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)

    # Get PVs
    # First determine amount to rotate second session
    if not best_rot:
        rot_deg = 0
    elif best_rot:
        _, rot_degs, _ = get_best_rot(mouse, arena1, day1, arena2, day2, pf_file=pf_file)
        if corr_type == 'us':
            rot_deg = rot_degs[0]
        elif corr_type == 'sm':
            rot_deg = rot_degs[1]

    if arena1 == arena2 and not best_rot:  # Don't resize 2nd session PFs if not rotating or looking b/w arenas
        pf1_shape = None
    else:  # Get dimensions of PFs in 1st session for resizing 2nd session to match if necessary
        try:
            PF1 = pf.load_pf(mouse, arena1, day1, pf_file=pf_file)
        except FileNotFoundError:
            raise RuntimeError('No placefields file found - can''t create 2d population vector')
        pf1_shape = PF1.tmap_sm[0].shape

    PV1us, PV1sm = pf.get_PV2(mouse, arena1, day1, speed_thresh=speed_thresh, pf_file=pf_file)
    PV2us, PV2sm = pf.get_PV2(mouse, arena2, day2, speed_thresh=speed_thresh, pf_file=pf_file,
                              rot_deg=rot_deg, resize_dims=pf1_shape)

    # Fill in as nan if session data is missing
    if (np.isnan(PV1us).all() and np.isnan(PV1sm).all()) or (np.isnan(PV2us).all() and np.isnan(PV2sm).all()):
        PV2dcorr_all, PV2dcorr_both = np.nan, np.nan
    else:  # Get the correct maps (smoothed vs. unsmoothed)

        if corr_type == 'sm':
            PV1, PV2 = PV1sm, PV2sm
        elif corr_type == 'us':
            PV1, PV2 = PV1us, PV2us

        # Now register between sessions
        PV1all, PV2all, PV1both, PV2both = registerPV(PV1, PV2, neuron_map, reg_session, shuf_map=shuf_map)

        if perform_corr:  # Now flatten PV arrays and get ALL corrs and BOTH corrs
            PV2dcorr_all, all_p = sstats.spearmanr(PV1all.reshape(-1), PV2all.reshape(-1), nan_policy='omit')
            PV2dcorr_both, both_p = sstats.spearmanr(PV1both.reshape(-1), PV2both.reshape(-1), nan_policy='omit')
        else:  # Dump these registered PVs into arrays for easy later re-shuffling
            PV2dcorr_all = (PV1all, PV2all)
            PV2dcorr_both = (PV1both, PV2both)

    return PV2dcorr_all, PV2dcorr_both


def registerPV(PV1, PV2, neuron_map, reg_session, shuf_map=False):
    """

    :param PV1: 1-d or 2-d population vector of event rates for all neurons in session 1
    :param PV2: same as above for session 2
    :param neuron_map: map between sessions obtained from SharpWave/Tenaspis github code
    :param reg_session: 2nd session id in session_list
    :return: PV1all, PV2allreg: PVs for ALL neurons active in EITHER session registered to each other -
             includes silent/new cells
             PV1both, PV2bothreg: PVs for neurons active in BOTH sessions registered to each other -
             no silent/new cells
    """

    # Identify if you are working with a 1d or 2d population vector
    if len(PV1.shape) == 1:
        type = '1d'
    elif len(PV1.shape) == 2:
        type = '2d'
        _, nbins = PV1.shape  # get #spatial bins

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
    if type == '1d':
        PV1all = np.concatenate((PV1, np.zeros(nnew)))
        PV2allreg = np.concatenate((np.zeros(PV1.shape), np.zeros(nnew)))
    elif type == '2d':
        # Find out where there is zero occupancy (e.g. nan in a spatial bin across all neurons) and
        # add that onto the end of the PV arrays
        pfpad1 = np.zeros((1, nbins))
        pf1nanbool = np.all(np.isnan(PV1), axis=0)
        pfpad1[0, pf1nanbool] = np.nan * np.ones(pf1nanbool.sum())
        PV1all = np.concatenate((PV1, np.matlib.repmat(pfpad1, nnew, 1)))

        pfpad2 = np.zeros((1, nbins))
        pf2nanbool = np.all(np.isnan(PV2), axis=0)
        pfpad2[0, pf2nanbool] = np.nan * np.ones(pf2nanbool.sum())
        PV2allreg = np.matlib.repmat(pfpad2, PV1.shape[0] + nnew, 1)
    PV2allreg[good_map_ind] = PV2[good_map]  # Dump in all neurons from second session that match those in first session.
    PV2allreg[len(PV1):] = PV2[new_ind]  # Add in new neurons from second session to end - needs checking!

    # Construct PVs with ONLY neurons detected in BOTH sessions
    PV1both = PV1[good_map_ind]
    PV2bothreg = PV2[good_map]

    return PV1all, PV2allreg, PV1both, PV2bothreg


def get_all_PV1corrs(mouse, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0, batch_map_use=False):
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
                    PVall, PVboth = PV1_corr_bw_sesh(mouse, arena1, day1, arena2, day2, batch_map_use=batch_map_use)
                    corrs_both[id1, id2] = PVboth
                    corrs_all[id1, id2] = PVall
                    shuf_all[id1, id2], shuf_both[id1, id2] = PV1_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf)
                except (FileNotFoundError, IndexError):
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


def get_all_PV2d_corrs(mouse, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0, batch_map_use=False, best_rot=False):
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
                    PVall, PVboth = PV2_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                                                     batch_map_use=batch_map_use, best_rot=best_rot)
                    corrs_both[id1, id2] = PVboth
                    corrs_all[id1, id2] = PVall
                    shuf_all[id1, id2], shuf_both[id1, id2] = PV2_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf)
                except (FileNotFoundError, RuntimeError, IndexError):
                    print('File Not Found for ' + mouse + ': ' + arena1 + str(day1) + ' to ' + arena2 + str(day2))

    return corrs_all, corrs_both, shuf_all, shuf_both


def PV2_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf, batch_map=True, debug=False):
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
    save_name = 'PV2shuf_corrs_' + arena2 + str(day2) + '_nshuf_' + str(nshuf) + '.pkl'
    dir_use = get_dir(mouse, arena1, day1)
    save_file = path.join(dir_use, save_name)

    if not path.exists(save_file):
        print('Getting shuffled 2-d PV corrs')
        # First get the population vectors and register them to one another.
        PVall, PVboth = PV2_corr_bw_sesh(mouse, arena1, day1, arena2, day2, shuf_map=True,
                                         batch_map_use=batch_map, perform_corr=False)
        try:
            PV1all, PV2all = PVall[0], PVall[1]
            nall = PV1all.shape[0]
            PV1both, PV2both = PVboth[0], PVboth[1]
            nboth = PV1both.shape[0]

            # Eiminate ALL spatial bins with no occupancy in either sesison to (hopefully) cut down on computation time below...
            good_bins = ~np.all(np.isnan(PV1both) | np.isnan(PV2both), axis=0)
            PV1all, PV1both = PV1all[:, good_bins], PV1both[:, good_bins]
            PV2all, PV2both = PV2all[:, good_bins], PV2both[:, good_bins]

            # Now shuffle things up and calculate!
            for n in tqdm(np.arange(nshuf)):
                corr_all, all_p = sstats.spearmanr(PV1all.reshape(-1),
                                                       PV2all[np.random.permutation(nall)].reshape(-1),
                                                       nan_policy='omit')
                corr_both, both_p = sstats.spearmanr(PV1both.reshape(-1),
                                                         PV2both[np.random.permutation(nboth)].reshape(-1),
                                                         nan_policy='omit')

                temp_all.append(corr_all)
                temp_both.append(corr_both)

            shuf_all = np.asarray(temp_all)
            shuf_both = np.asarray(temp_both)

            # pickle data for later use
            dump([['mouse', 'arena1', 'day1', 'arena2', 'day2', 'shuf_all', 'shuf_both', 'nshuf'],
                  [mouse, arena1, day1, arena2, day2, shuf_all, shuf_both, nshuf]],
                 open(save_file, 'wb'))
        except TypeError:
            print('Error somewhere along the line - sending all shuffled corrs to NaNs')
            shuf_all, shuf_both = np.nan, np.nan
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
                    rot_deg=0, shuf_map=False, debug=False, batch_map_use=True, speed_threshold=True,
                    keep_poor_overlap=False):
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

    # Eliminate neurons that are only active during immobility - will encounter errors trying to calculate correlations
    # using their all-zero transient maps.
    if speed_threshold:
        good_neurons_base, good_neurons_reg = eliminate_immobile_neurons(PF1.PSAboolrun, PF2.PSAboolrun,
                                                                         valid_neurons_base, valid_neurons_reg)
    elif not speed_threshold:
        good_neurons_base, good_neurons_reg = valid_neurons_base, valid_neurons_reg

    # Shuffle mapping between sessions if specified
    if shuf_map:
        good_neurons_reg = np.random.permutation(good_neurons_reg)
    ngood = len(good_neurons_base)

    # Step through each mapped neuron and get corrs between each
    rot = int(rot_deg/90)
    no_run_events_bool = np.ones(ngood, dtype=bool)

    # First get rid of any non-validly mapped neurons
    tmaps1_us_valid = [PF1.tmap_us[neuron] for neuron in good_neurons_base]
    tmaps1_sm_valid = [PF1.tmap_sm[neuron] for neuron in good_neurons_base]
    tmaps2_us_valid = [PF2.tmap_us[rneuron] for rneuron in good_neurons_reg]
    tmaps2_sm_valid = [PF2.tmap_sm[rneuron] for rneuron in good_neurons_reg]

    # Next rotate session 2 maps if specified
    if rot_deg != 0:
        tmaps2_us_rot = rotate_tmaps(tmaps2_us_valid, rot_deg)
        tmaps2_sm_rot = rotate_tmaps(tmaps2_sm_valid, rot_deg)
    elif rot_deg == 0:
        tmaps2_us_rot, tmaps2_sm_rot = tmaps2_us_valid, tmaps2_sm_valid

    # Next, reshape tmaps from session 2 if doing across arena comparisons
    if rot == 0 and arena1 == arena2:
        tmaps2_us_use, tmaps2_sm_use = tmaps2_us_rot, tmaps2_sm_rot
    else:
        tmap1_shape = tmaps1_us_valid[0].shape
        tmaps2_us_use = pf.rescale_tmaps(tmaps2_us_rot, tmap1_shape)
        tmaps2_sm_use = pf.rescale_tmaps(tmaps2_sm_rot, tmap1_shape)

    # Finally do your correlations!
    corrs_us = get_pf_corrs(tmaps1_us_valid, tmaps2_us_use, keep_poor_overlap=keep_poor_overlap)
    corrs_sm = get_pf_corrs(tmaps1_sm_valid, tmaps2_sm_use, keep_poor_overlap=keep_poor_overlap)

    # Old, overly dense code below
    corrs_us_old, corrs_sm_old = [], []  # Initialize correlation lists
    for base_neuron, reg_neuron in zip(good_neurons_base, good_neurons_reg):

        # if debug and base_neuron == 364:  # for debugging nans in sstats.spearmanr
        try:
            if rot == 0 and arena1 == arena2:  # Do correlations directly if possible
                corr_us, p_us, poor_overlap_us = spearmanr_nan(PF1.tmap_us[base_neuron].reshape(-1),
                                                               PF2.tmap_us[reg_neuron].reshape(-1))

                corr_sm, p_sm, poor_overlap_sm = spearmanr_nan(PF1.tmap_sm[base_neuron].reshape(-1),
                                                               PF2.tmap_sm[reg_neuron].reshape(-1))

            else:  # rotate and resize PF2 before doing corrs if rotations are specified
                PF1_size = PF1.tmap_us[0].shape
                corr_us, p_us, poor_overlap_us = spearmanr_nan(PF1.tmap_us[base_neuron].reshape(-1),
                                                               np.reshape(sk_resize(np.rot90(PF2.tmap_us[reg_neuron], rot),
                                                                                    PF1_size, anti_aliasing=True), -1))

                corr_sm, p_sm, poor_overlap_sm = spearmanr_nan(np.reshape(PF1.tmap_sm[base_neuron], -1),
                                                               np.reshape(sk_resize(np.rot90(PF2.tmap_sm[reg_neuron], rot),
                                                                                    PF1_size, anti_aliasing=True), -1))
        except RuntimeWarning:  # Note you will have to enable warnings for this to work a la >> import warnings, >>warnings.filterwarnings('error', category=RuntimeWarning)
            print('RunTimeWarning Encountered in some basic scipy/numpy functions - should probably debug WHY this is happening')
            print('Base_neuron = ' + str(base_neuron))

        # exclude any correlations that would throw a scipy.stats.spearmanr RuntimeWarning due to
        # # poor overlap after rotation...
        if keep_poor_overlap: # This is necessary to make pfscroll work
            corrs_us_old.append(corr_us)
            corrs_sm_old.append(corr_sm)
        elif not keep_poor_overlap:
            if not poor_overlap_us: corrs_us_old.append(corr_us)
            if not poor_overlap_sm: corrs_sm_old.append(corr_sm)

    corrs_us_old, corrs_sm_old = np.asarray(corrs_us_old), np.asarray(corrs_sm_old)

    return corrs_us, corrs_sm


def eliminate_immobile_neurons(PSAboolrun1, PSAboolrun2, valid_neurons1, valid_neurons2):
    """Eliminate neurons that are only active during immobility - will encounter errors trying to calculate
    correlations using their all-zero transient maps."""

    # Identify mapped neurons with least one calcium event after speed thresholding
    run_events_bool = (PSAboolrun1[valid_neurons1, :].sum(axis=1) > 0) & \
                      (PSAboolrun2[valid_neurons2, :].sum(axis=1) > 0)

    # Refine map again to only include active neurons after speed thresholding
    good_neurons_base = valid_neurons1[run_events_bool].astype(np.int64)
    good_neurons_reg = valid_neurons2[run_events_bool].astype(np.int64)

    return good_neurons_base, good_neurons_reg

def get_pf_corrs(tmaps1, tmaps2, keep_poor_overlap=False):
    """
    get correlations between matching transient event maps in lists
    :param tmaps1: list of 2d or 1d transient event maps from 1st session, each an ndarray
    :param tmaps2: list of event maps for session2 for the same neurons as in tmaps1
    :param keep_poor_overlap: boolean, True = keep all corrs, even if Nan, False(default) = exclude those where the
    animal does not occupy any of the same spatial bins from session 1 to session 2
    :return: ndarray of spearman correlations
    """

    corrs = []
    for tmap1, tmap2 in zip(tmaps1, tmaps2):
        try:
            corr, p, poor_overlap = spearmanr_nan(tmap1.reshape(-1), tmap2.reshape(-1))

        except RuntimeWarning:  # Note you will have to enable warnings for this to work a la >> import warnings, >>warnings.filterwarnings('error', category=RuntimeWarning)
            print('RunTimeWarning Encountered in some basic scipy/numpy functions in ""get_pf_corrs" - should probably debug WHY this is happening')

        # exclude any correlations that would throw a scipy.stats.spearmanr RuntimeWarning due to
        # # poor overlap after rotation...
        if keep_poor_overlap: # This is necessary to make pfscroll work
            corrs.append(corr)
        elif not keep_poor_overlap:
            if not poor_overlap:
                corrs.append(corr)
            # elif poor_overlap:
                # print('debugging get_pf_corrs')

    return np.asarray(corrs)


def rotate_tmaps(tmaps, rot_deg):
    """
        Rotate all transient maps in tmaps in 90 degree increments
        :param tmaps: list of tmaps
        :param rot_deg: int, # degrees to rotate map, must be 90 degree increments
        :return: list of rotated tmaps
        """
    rot = int(rot_deg / 90)
    tmaps_rot = []
    for tmap in tmaps:
        tmaps_rot.append(np.rot90(tmap, rot))

    return tmaps_rot


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


def get_best_rot(mouse, arena1='Shock', day1=-2, arena2='Shock', day2=-1,
                 pf_file='placefields_cm1_manlims_1000shuf.pkl', batch_map_use=False):
    """
    Gets the rotation of the arena in day2 that produces the best correlation. Will load previous runs from file
    saved in the appropriate directory by default
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
                corrs_us, corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2, pf_file=pf_file,
                                                     rot_deg=rot, shuf_map=False, batch_map_use=batch_map_use)
            except IndexError:  # Fix for missing sessions
                print('Index Error for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                corrs_us = corrs_sm = np.ones(1)*np.nan
            corr_mean_all[0, idr] = corrs_us.mean(axis=0)
            corr_mean_all[1, idr] = corrs_sm.mean(axis=0)

        corr_mean_all[np.isnan(corr_mean_all)] = -1  # Set all nan-values to -1 (occurs when there is NO overlap bw occupancy).
        best_rot = np.array(rots)[corr_mean_all.argmax(axis=1)]
        best_corr_mean = corr_mean_all.max(axis=1)

        # Pickle results
        dump([['Mouse', '[Arena1, Arena2]', 'day1', 'day2', 'best_corr_mean[un-smoothed, smoothed]',
               'best_rot[un-smoothed, smoothed]',
               'corr_mean_all[un-smoothed, smoothed]'],
              [mouse, [arena1, arena2], day1, day2, best_corr_mean, best_rot, corr_mean_all]],
             open(save_file, "wb"))
    elif path.exists(save_file):  # Load previous run and let user know
        print('Loading previous 2d placefield analysis for ' + mouse + ' ' + arena1 + ' day ' + str(day1) +
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


def plot_pfcorr_bygroup(corr_mean_mat, arena1, arena2, group_type, shuf_mat=None, save_fig=True,
                        color='b', ax_use=None, offset=0, group_desig=1, best_rot=False,
                        prefix='PFcorrs', linetype='-'):
    """
    Scatterplot of correlations before shock, after, and several other groupings
    :param corr_mean_mat: nmice x 7 x 7 array of mean corr values for each mouse
    :param arena1: 'Shock' or 'Neutral'
    :param arena2:
    :param group_type: e.g. 'Control' or 'Anisomycin'
    :param shuf_mat: if supplied, calculated 95% CIs and plots on figure.
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

    # Plot data for each epoch
    for epoch_num in unique_epochs:
        corr_means.append(np.nanmean(corr_mean_mat[epochs == epoch_num]))
    axl = ax.plot(unique_epochs, corr_means, linetype, color=color)

    # Plot CIs if shuffled data provided
    aCI = None
    CIs = []
    if shuf_mat is not None:
        nshuf = shuf_mat.shape[3]
        for epoch_num in unique_epochs:
            # CIs.append(mean_CI(shuf_mat.reshape(-1, nshuf)[epochs.reshape(-1) == epoch_num]))
            CIs.append(get_CI(np.nanmean(shuf_mat.reshape(-1, nshuf)[epochs.reshape(-1) == epoch_num], axis=0)))
        aCI = ax.plot(np.matlib.repmat(unique_epochs, 3, 1).transpose(), np.asarray(CIs), 'k--')
        aCI[1].set_linestyle('-')
        [a.set_color([0, 0, 0, 0.5]) for a in aCI]


    if save_fig:
        fig.savefig(path.join(err.pathname, prefix + ' ' + arena1 + ' v '
                    + arena2 + ' ' + group_type + 'group_desig' + str(group_desig) + 'best_rot' + str(best_rot) +
                    '.pdf'))

    return fig, ax, ascat, axl, aCI


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
    pairs[0, 1] = 0  # Day -2 v -1
    pairs[1, 3] = 1  # -1 v 4hr
    pairs[3, 4] = 2  # 4hr v 1
    pairs[4, 5] = 3  # 1 v 2
    pairs[5, 6] = 4  # 2 v 7
    # pair_ids, grps = [0, 1, 3, 4, 5], [0, 1, 2, 3, 4]
    # for ind, idd in enumerate(pair_ids):
    #     pairs[idd, idd+1] = grps[ind]

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
    least one nan (i.e. the mouse never occupied any of the bins in the second session that it did in the first session).
    Note that this will also tmaps where a neuron had no transients above the running threshold.

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


def get_group_PV1d_corrs(mice, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0, batch_map_use=True):
    """
    Assembles a nice matrix of mean correlation values between 1d PVs on days/arenas specified.
    :param mice:
    :param arena1:
    :param arena2:
    :param days:
    :param nshuf:
    :param batch_map_use
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
            get_all_PV1corrs(mouse, arena1, arena2, days, nshuf=nshuf, batch_map_use=batch_map_use)

    return PV1_all_all, PV1_both_all, PV1_both_shuf, PV1_all_shuf


def get_group_PV2d_corrs(mice, arena1, arena2, days=[-2, -1, 0, 4, 1, 2, 7], nshuf=0, batch_map_use=True,
                         best_rot=False):
    """
    Assembles a nice matrix of mean correlation values between 1d PVs on days/arenas specified.
    :param mice:
    :param arena1:
    :param arena2:
    :param days:
    :param nshuf:
    :param batch_map_use:
    :param best_rot:
    :param pf_file:
    :return:
    """

    # pre-allocate
    ndays = len(days)
    nmice = len(mice)
    PV2d_both_all, PV2d_all_all = np.ones((nmice, ndays, ndays))*np.nan, np.ones((nmice, ndays, ndays))*np.nan
    PV2d_both_shuf, PV2d_all_shuf = np.ones((nmice, ndays, ndays, nshuf))*np.nan, np.ones((nmice, ndays, ndays, nshuf))*np.nan

    for idm, mouse in enumerate(mice):
        PV2d_all_all[idm, :, :], PV2d_both_all[idm, :, :], PV2d_both_shuf[idm, :, :, :], PV2d_all_shuf[idm, :, :, :] = \
            get_all_PV2d_corrs(mouse, arena1, arena2, days, nshuf=nshuf, batch_map_use=batch_map_use, best_rot=best_rot)

    return PV2d_all_all, PV2d_both_all, PV2d_both_shuf, PV2d_all_shuf


class PlaceFieldHalf:
    """Class to visualize and quantify wihin-session stability
    :param: mouse, arena day: self-explanatory
    :param: nshuf: #unit-id and circular event-train shuffles to perform when calculating chance-level between half
    correlations
    :param: type: "half" (default) calculates 1st v 2nd half, "odd/even" calculates odd v even minutes. Can also take a
    list of 3 values to calulate between arbitrary parts of the session of the form [numerator1, numerator2, demoninator].
    e.g. [1, 3, 5] would calculate stability between the first and third 1/5s of the session.
    :param: quickload: True = load in correlations only (faster), False (default) = load in all placefields
    :param: can take other kwarg inputs for PlaceFields.placefield, e.g. align_from_end=True...
    """
    def __init__(self, mouse, arena, day, nshuf=100, plot_type="half", quickload=False, **kwargs):

        self.mouse = mouse
        self.arena = arena
        self.day = day
        self.ncircshuf = nshuf
        self.plot_type = plot_type
        self.generate_save_name()  # Get file name for loading in/saving later
        isrunning1, isrunning2 = None, None  # By default don't use arbitrary periods of the session
        if self.plot_type == "half":
            half1, half2 = 1, 2
        elif self.plot_type == "odd/even" or self.plot_type == "even/odd":
            half1, half2 = "odd", "even"
        elif type(self.plot_type) in [list, tuple]:
            half1, half2 = None, None
            isrunning1 = self.generate_arbitrary_bool(self.plot_type[0], self.plot_type[2])
            isrunning2 = self.generate_arbitrary_bool(self.plot_type[1], self.plot_type[2])

        try:  # load in existing file if there.
            self._load()
            if quickload:  # don't load in all placefields
                PF = pf.load_pf(mouse, arena, day)
                self.nneurons = len(PF.tmap_sm)
            elif not quickload:  # load in all placefields for later access
                self.PF1 = pf.placefields(mouse, arena, day, nshuf=0, half=half1, save_file=None,
                                          isrunning_custom=isrunning1, **kwargs)
                self.PF2 = pf.placefields(mouse, arena, day, nshuf=0, half=half2, keep_shuffled=False, save_file=None,
                                          isrunning_custom=isrunning2, **kwargs)
                self.nneurons = len(self.PF1.tmap_sm)
            # self.calc_half_corrs()  # don't calculate actual correlations directly - should already be loaded!
            self.tmap_sm_corrs = self.half_corrs['tmap_sm_corrs']
            self.idshuf_sm_mean = self.half_corrs['idshuf_mean']
            self.circshuf_sm_mean = self.half_corrs['circshuf_sm_mean']
        except FileNotFoundError:
            try:
                # Create PF object for each half - only shuffle spike train in second half of session
                self.PF1 = pf.placefields(mouse, arena, day, nshuf=0, half=half1, save_file=None,
                                          isrunning_custom=isrunning1, **kwargs)
                self.PF2 = pf.placefields(mouse, arena, day, nshuf=nshuf, half=half2, keep_shuffled=True,
                                          save_file=None, isrunning_custom=isrunning2, **kwargs)
                self.nneurons = len(self.PF1.tmap_sm)

                # Get correlations between 1st and 2nd half
                self.calc_half_corrs()

                # Calculate chance level
                self.calc_idshuffled_corrs(nidshuf=nshuf)
                self.calc_circshuffled_corrs()
                self._save()

                # Now load everything into a nice dict for immediate use
                self._load()
            except FileNotFoundError: # (FileNotFoundError, ValueError, AttributeError):
                self.half_corrs = {}  # Make empty if not you can't load in

    def generate_arbitrary_bool(self, numerator, demoninator):
        """Generate a boolean to calculate place-fields for an abitrary part of the session. E.g numerator=3 and
        demoninator=5 would generate the place-field activity during the 3rd fifth of the session."""
        isrunning = pf.get_running_bool(self.mouse, self.arena, self.day)  # get times when mouse is running
        nframes_period = np.floor(len(isrunning) / demoninator).astype('int')  # identify # frames in each period of the session

        # Keep only legitimate peiod from isrunning boolean array
        isrunning_custom = np.zeros_like(isrunning).astype(bool)
        isrunning_custom[((numerator-1)*nframes_period):(numerator*nframes_period)] = \
            isrunning[((numerator-1)*nframes_period):(numerator*nframes_period)]

        return isrunning_custom

    def pfscroll(self):
        """Scroll through 1st and 2nd half placefields simultaneously"""
        self.PF1.pfscroll()
        if plt.get_backend() == 'Qt5Agg':
            plt.get_current_fig_manager().window.setGeometry(145, 45, 1245, 420)
        else:
            self.PF1.f.fig.set_size_inches([12.4, 3.6])
        self.PF2.pfscroll(link_PFO=self.PF1.f)
        if plt.get_backend() == 'Qt5Agg':
            plt.get_current_fig_manager().window.setGeometry(145, 525, 1245, 420)
        else:
            self.PF2.f.fig.set_size_inches([12.4, 3.6])

    def calc_half_corrs(self):
        """Calculate placefield correlations between 1st and 2nd half of session"""
        self.tmap_us_corrs = get_pf_corrs(self.PF1.tmap_us, self.PF2.tmap_us)
        self.tmap_sm_corrs = get_pf_corrs(self.PF1.tmap_sm, self.PF2.tmap_sm)

    def calc_idshuffled_corrs(self, nidshuf=1000, ax=False):
        """Calculate mean correlations between placefields after shuffling unit id between halves.
        Set ax to True or an existing figure axes to plot results"""
        # nneurons = len(self.tmap_us_corrs)
        self.nidshuf = nidshuf

        def mean_idshuf_corrs(tmaps1, tmaps2, shuf_ids):
            """sub-function to calculately mean shuffled correlations in one-line below """
            tmaps2_shuf = [tmaps2[id] for id in shuf_ids]
            shuf_corrs = get_pf_corrs(tmaps1, tmaps2_shuf)

            return np.mean(shuf_corrs)

        idshuf_us_mean, idshuf_sm_mean = [], []
        for shuf in tqdm(range(nidshuf)):
            shuf_ids = np.random.permutation(self.nneurons)  # shuffle unit ids
            idshuf_sm_mean.append(mean_idshuf_corrs(self.PF1.tmap_sm, self.PF2.tmap_sm, shuf_ids))
            idshuf_us_mean.append(mean_idshuf_corrs(self.PF1.tmap_us, self.PF2.tmap_us, shuf_ids))

        self.idshuf_us_mean = np.asarray(idshuf_us_mean)
        self.idshuf_sm_mean = np.asarray(idshuf_sm_mean)

        if ax:
            if type(ax) == bool:  # create figure axes if none specified
                _, ax = plt.subplots()
            self.plot_shuffled_corrs(ax, self.tmap_sm_corrs.mean(), self.idshuf_sm_mean,
                                     names=['Data', 'Id Shuffle'])
            # with sns.axes_style('white'):
            #     sns.set_context('notebook')
            #     if not isinstance(ax, axes.Axes):
            #         fig, ax = plt.subplots()
            #     sns.histplot(self.idshuf_sm_mean, ax=ax)
            #     sns.despine()
            #     ax.axvline(self.tmap_sm_corrs.mean(), color='k', linestyle='--')
            #     ax.set_xlabel('Mean Spearman Corr')
            #     ax.legend(['Data', 'Shuffled'])
            #     ax.set_title('1st vs 2nd half')

    def calc_circshuffled_corrs(self, ax=False):
        """Calculate and plot correlations where spike trains have been shuffled in second half of session"""
        circshuf_corrs = np.ones((self.nneurons, self.ncircshuf))*np.nan
        print('Calculating circularly shuffled correlations')
        for idn in tqdm(range(self.nneurons)):
            circcorrs, _, _ = zip(*[spearmanr_nan(self.PF1.tmap_sm[idn].reshape(-1), pf2shuf.reshape(-1)) for pf2shuf in
                             self.PF2.tmap_sm_shuf[idn]])
            circshuf_corrs[idn, :] = np.asarray(circcorrs)

        self.circshuf_sm_mean = np.nanmean(circshuf_corrs, axis=0)

        if ax:
            if type(ax) == bool:  # create figure axes if none specified
                _, ax = plt.subplots()
            self.plot_shuffled_corrs(ax, self.tmap_sm_corrs.mean(), self.circshuf_sm_mean,
                                     names=['Data', 'Circ Shuffle'])


    def plot_shuffled_corrs(self, ax, corrs_mean, shuf_mean_corrs, names=['Data', 'Shuffled']):
        with sns.axes_style('white'):
            sns.set_context('notebook')
            if not isinstance(ax, axes.Axes):
                fig, ax = plt.subplots()
            sns.histplot(shuf_mean_corrs, ax=ax)
            sns.despine()
            ax.axvline(corrs_mean, color='k', linestyle='--')
            ax.set_xlabel('Mean Spearman Corr')
            ax.legend(names)
            ax.set_title('1st vs 2nd half')

        return ax

    def generate_save_name(self):
        if self.plot_type == "half":
            name_append = ""
        elif self.plot_type == "odd/even" or self.plot_type == "even/odd":
            name_append = "odd_v_even"
        elif type(self.plot_type) in [list, tuple]:
            name_append = '_' + str(self.plot_type[0]) + '_' + str(self.plot_type[1]) + '_' + str(self.plot_type[2])
        self.save_name = path.join(sd.find_eraser_session(self.mouse, self.arena, self.day)['Location'],
                                   "pfhalfcorrs_" + str(self.ncircshuf) + 'shuf' + name_append + '.pkl')

    def _save(self):

        half_corrs = {'mouse': self.mouse, 'arena': self.arena, 'day': self.day, 'ncircshuf': self.ncircshuf,
                      'nidshuf': self.nidshuf, 'idshuf_mean': self.idshuf_sm_mean,
                      'circshuf_sm_mean': self.circshuf_sm_mean, 'tmap_sm_corrs': self.tmap_sm_corrs}
        with open(self.save_name, 'wb') as f:
            dump(half_corrs, f)

    def _load(self):
        # save_name = path.join(sd.find_eraser_session(self.mouse, self.arena, self.day)['Location'],
        #                       "pfhalfcorrs_" + str(self.ncircshuf) + 'shuf.pkl')
        with open(self.save_name, 'rb') as f:
            self.half_corrs = load(f)


class SessionStability:
    """Class to easily look at within session stability across all mice/sessions"""
    def __init__(self, plot_type='half'):
        """Construct dictionary with each mouse group broken down into Shock/Open mean between-half correlations
        and 95% CIs
        :param: plot_type = 'half' or 'odd_v_even" to calculate correlations based on 1st v 2nd half or odd v even minutes
        """
        self.mice = {'learners': err.learners, 'nonlearners': err.nonlearners, 'ani': err.ani_mice_good}
        self.plot_type = plot_type
        # self.amice = err.ani_mice_good
        # self.lmice = err.learners
        # self.nlmice = err.nonlearners
        self.arenas = ['Open', 'Shock']
        self.days = [-2, -1, 0, 4, 1, 2, 7]
        self.half_corrs = {}
        for item in self.mice.items():
            group_name, mouse_list = item[0], item[1]
            self.half_corrs[group_name] = {}
            for arena in self.arenas:
                self.half_corrs[group_name][arena] = {}
                session_corrs, circshufCI, idshufCI = [], [], []
                for day in self.days:
                    mouse_corrs, circ_shuf, id_shuf = [], [], []
                    if not (day == 0 and arena == "Shock"):  # calculate for everything except shock day 0!
                        for mouse in mouse_list:
                            # First check to see if "ALIGNTOEND" is in the working directory and align data from end
                            dir_use = get_dir(mouse, arena, day)
                            align_end_bool = str(dir_use).upper().find('ALIGNTOEND') != -1  # set to True if "aligntoend" is in folder name, False otherwise
                            quickload = not align_end_bool  # don't quickload data below if special alignment is needed
                            pfh = PlaceFieldHalf(mouse=mouse, arena=arena, day=day, quickload=True, plot_type=plot_type,
                                                 align_from_end=align_end_bool)
                            if 'tmap_sm_corrs' in pfh.half_corrs:
                                mouse_corrs.append(pfh.half_corrs['tmap_sm_corrs'].mean())
                            else:
                                mouse_corrs.append(np.nan)
                            if 'circshuf_sm_mean' in pfh.half_corrs:
                                circ_shuf.append(pfh.half_corrs['circshuf_sm_mean'])
                            if 'idshuf_mean' in pfh.half_corrs:
                                id_shuf.append(pfh.half_corrs['idshuf_mean'])
                    elif day == 0 and arena == "Shock":  # Don't calculate anything for shock day 0!
                        for mouse in mouse_list:  # send everything to nans!
                            mouse_corrs.append(np.nan)
                            circ_shuf.append(np.nan)
                            id_shuf.append(np.nan)

                    # Now append values to list above and calculate 95% CIs from mean shuffled data
                    session_corrs.append(mouse_corrs)
                    circshufCI.append(get_CI(np.asarray(circ_shuf).mean(axis=0)))
                    idshufCI.append(get_CI(np.asarray(id_shuf).mean(axis=0)))

                self.half_corrs[group_name][arena]['circshufCI'] = np.asarray(circshufCI)
                self.half_corrs[group_name][arena]['idshufCI'] = np.asarray(idshufCI)
                self.half_corrs[group_name][arena]['tmap_sm_mean'] = np.asarray(session_corrs)

    def plot_stability(self, group, CI='circshufCI', colorby='arena', bw_arena=False):
        """Plot within session stability for all mice in designated group"""
        narenas = len(self.arenas)  # is this used?

        # set up figure
        fig, ax = plt.subplots(2, 1)
        sns.despine(fig=fig)  # remove top and right axes
        fig.set_size_inches([11, 7.5])

        # set up colors
        if colorby == 'arena':
            colors = ['k', 'r']
        elif colorby == 'group':
            colors = [color_dict[group.lower()], color_dict[group]]

        # Stagger points to one side when plotting with stripplot
        def offset_points(ax, offset, collection_range):
            for idc in collection_range:
                offset_data = ax.collections[idc].get_offsets().data + \
                              np.matlib.repmat([offset, 0],
                                               ax.collections[2].get_offsets().data.shape[0], 1)
                ax.collections[idc].set_offsets(offset_data)

        # now plot everything
        offsets = [-0.125, 0.125]  # for offsetting data plotted side-by-side
        ncollections = [0]
        for ida, arena in enumerate(self.arenas):
            # set up axes
            if bw_arena: # plot on same subplot for between arena plots
                ax_use = ax[0]
            elif not bw_arena: # otherwise plot separately
                ax_use = ax[ida]

            data_use = self.half_corrs[group][arena]  # pull out correct group

            # plot data
            sns.stripplot(data=data_use['tmap_sm_mean'].swapaxes(0, 1), color=colors[ida], ax=ax_use)
            nsessions = data_use['tmap_sm_mean'].shape[0]
            if bw_arena:  # offset points!
                ncollections.append(len(ax_use.collections))
                collection_range = range(ncollections[-1] - nsessions, ncollections[-1])
                offset_points(ax_use, offsets[ida], collection_range)

            # Fill in 95% CIs for shuffled data
            plt.sca(ax_use)
            plt.fill_between(range(data_use[CI].shape[0]), data_use[CI][:, 0],
                             data_use[CI][:, 2], color=colors[ida], alpha=0.3)  # 95% CIs
            plt.plot(range(data_use[CI].shape[0]), data_use[CI][:, 1], color=colors[ida], alpha=0.3)  # mean shuffled

            # Label everything
            ax_use.set_xticklabels([str(day) for day in self.days])
            if ida == 1:
                ax_use.set_xlabel('Session')
            if self.plot_type == 'half':
                ax_use.set_title(group.capitalize() + ': ' + arena + ' 1st v 2nd Half')
            elif self.plot_type == 'odd/even' or self.plot_type == 'even/odd':
                ax_use.set_title(group.capitalize() + ': ' + arena + ' Odd v Even Minutes')
            ax[ida].set_ylabel('Mean Spearman Correlation')
            if ida == 1 and not bw_arena:
                plt.legend(['CI', 'Data'])
            elif ida == 1 and bw_arena:
                plt.legend([ax_use.collections[0], ax_use.collections[ncollections[1] + 1]],
                           ['Open', 'Shock'])

        # connect dots if bw_arena plots! need vetting!
        if bw_arena:
            collection1 = ax_use.collections[(ncollections[1] - nsessions):ncollections[1]]  # Grab arena 1 points
            collection2 = ax_use.collections[(ncollections[2] - nsessions):ncollections[2]]  # Grab arena 2 points

            # iterate through each session and connect the dots
            for col1, col2 in zip(collection1, collection2):
                if type(col1.get_offsets()) is np.ma.core.MaskedArray:
                    xvals = np.asarray([col1.get_offsets().data[:, 0],
                                        col2.get_offsets().data[:, 0]])
                    yvals = np.asarray([col1.get_offsets().data[:, 1],
                                        col2.get_offsets().data[:, 1]])
                elif type(col1.get_offsets()) is np.ndarray:
                    xvals = np.asarray([col1.get_offsets()[:, 0],
                                        col2.get_offsets()[:, 0]])
                    yvals = np.asarray([col1.get_offsets()[:, 1],
                                        col2.get_offsets()[:, 1]])
                ax_use.plot(xvals, yvals, 'k', alpha=0.3)



## Object to map and view placefields for same neuron mapped between different sessions
class PFCombineObject:
    """map and view placefields for same neuron mapped between different sessions"""
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
        good_map_ind = np.where(good_map_bool)[0]
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

        # Get correlations between sessions! Note these are not speed-thresholded (quick bug fix).
        self.corrs_us, self.corrs_sm = pf_corr_bw_sesh(mouse, arena1, day1, arena2, day2,
                                                       pf_file=pf_file, debug=debug, speed_threshold=False,
                                                       keep_poor_overlap=True)

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


## Create class to calculate and save correlations between sessions with neuron_map shuffled
class ShufMap:
    """Create class to calculate and save correlations between sessions with neuron_map shuffled (basically
    a unit id shuffle)"""
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
    """construct and keep all group placefield data in a nice format and plot things..."""
    def __init__(self):
        self.amice = err.ani_mice_good
        self.lmice = err.learners
        self.nlmice = err.nonlearners
        self.days = [-2, -1, 0, 4, 1, 2, 7]

    def _save(self, dir=r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser'):
        dump(self.data, open(path.join(dir, 'group_data_rot=' + str(self.best_rot) + '_batch_map=' +
                                       str(self.batch_map) + '.pkl'), 'wb'))
        return None

    def _load(self, dir=r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser', best_rot=True, batch_map=True):
        self.data = load(open(path.join(dir, 'group_data_rot=' + str(best_rot) + '_batch_map=' +
                                        str(batch_map) + '.pkl'), 'rb'))
        self.best_rot = best_rot
        self.batch_map = batch_map

    def construct(self, types=['PFsm', 'PFus', 'PV1dboth', 'PV1dall', 'PV2dboth', 'PV2dall'], best_rot=True,
                  pf_file='placefields_cm1_manlims_1000shuf.pkl', nshuf=1000, batch_map=True):
        """Sets up all data in well-organized dictionary: data[type]['data' or 'shuf'][group][arena_type] where
        arena_type=0 for Open, 1 for Shock, and 2 for Open v Shock"""
        # perform PFcorrs at best rotation between session if True, False = no rotation
        groups = ['Learners', 'Nonlearners', 'Ani']
        # group_dict = dict.fromkeys(groups, {'corrs': [], 'shuf': []})
        self.data = dict.fromkeys(types)  # pre-allocate
        self.best_rot = best_rot
        self.nshuf = nshuf
        self.batch_map = batch_map
        self.cmperbin = pf.load_pf(self.lmice[0], 'Shock', -2, pf_file=pf_file).cmperbin
        self.nshuf2d = [1000, 1000, 0]  # for 'Shock', 'Open', and 'Open v Shock' until all shuffles run
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
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                batch_map_use=batch_map)
                    _, tempnl, _, temp_sh_nl = get_group_pf_corrs(self.nlmice, arena1, arena2, self.days,
                                                                  best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                    _, tempa, _, temp_sh_a = get_group_pf_corrs(self.amice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                batch_map_use=batch_map)
                elif type == 'PFus':
                    templ, _, temp_sh_l, _ = get_group_pf_corrs(self.lmice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                batch_map_use=batch_map)
                    tempnl, _, temp_sh_nl, _ = get_group_pf_corrs(self.nlmice, arena1, arena2, self.days,
                                                                  best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                    tempa, _, temp_sh_a, _ = get_group_pf_corrs(self.amice, arena1, arena2, self.days,
                                                                best_rot=best_rot, pf_file=pf_file, nshuf=nshuf,
                                                                batch_map_use=batch_map)
                elif type == 'PV1dboth':
                    _, templ, _, temp_sh_l = get_group_PV1d_corrs(self.lmice, arena1, arena2, self.days, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                    _, tempnl, _, temp_sh_nl = get_group_PV1d_corrs(self.nlmice, arena1, arena2, self.days, nshuf=nshuf,
                                                                    batch_map_use=batch_map)
                    _, tempa, _, temp_sh_a = get_group_PV1d_corrs(self.amice, arena1, arena2, self.days, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                elif type == 'PV1dall':
                    templ, _, temp_sh_l, _ = get_group_PV1d_corrs(self.lmice, arena1, arena2, self.days, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                    tempnl, _, temp_sh_nl, _ = get_group_PV1d_corrs(self.nlmice, arena1, arena2, self.days, nshuf=nshuf,
                                                                    batch_map_use=batch_map)
                    tempa, _, temp_sh_a, _ = get_group_PV1d_corrs(self.amice, arena1, arena2, self.days, nshuf=nshuf,
                                                                  batch_map_use=batch_map)
                elif type == 'PV2dboth':
                    _, templ, _, temp_sh_l = get_group_PV2d_corrs(self.lmice, arena1, arena2, self.days,
                                                                  nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                  batch_map_use=batch_map)
                    try:
                        _, tempnl, _, temp_sh_nl = get_group_PV2d_corrs(self.nlmice, arena1, arena2, self.days,
                                                                        nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                        batch_map_use=batch_map)
                    except TypeError:
                        print('Debugging PFGroup.construct()')
                        a = 1
                    _, tempa, _, temp_sh_a = get_group_PV2d_corrs(self.amice, arena1, arena2, self.days,
                                                                  nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                  batch_map_use=batch_map)
                elif type == 'PV2dall':
                    templ, _, temp_sh_l, _ = get_group_PV2d_corrs(self.lmice, arena1, arena2, self.days,
                                                                  nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                  batch_map_use=batch_map)
                    tempnl, _, temp_sh_nl, _ = get_group_PV2d_corrs(self.nlmice, arena1, arena2, self.days,
                                                                    nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                    batch_map_use=batch_map)
                    tempa, _, temp_sh_a, _ = get_group_PV2d_corrs(self.amice, arena1, arena2, self.days,
                                                                  nshuf=self.nshuf2d[ida], best_rot=best_rot,
                                                                  batch_map_use=batch_map)

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
            self.data['batch_map'] = batch_map

    def scatterbar_bw_days(self, type='PFsm', ax_use=None):
        # Set up plots
        fig, ax = self.figset(ax_use, nplots=[3, 1], size=[10.1, 9.2])
        save_flag = False  # Set up saving plots
        data_dict, shuf_dict = self.data[type]['data'], self.data[type]['shuf']
        titles = list(data_dict.keys())
        for idd, (data, shuf) in enumerate(zip(data_dict.values(), shuf_dict.values())):
            nmice = data[0].shape[0]
            pairs, labels = get_seq_time_pairs(nmice)

            # Plot data
            erp.scatterbar(data[0][~np.isnan(pairs)], pairs[~np.isnan(pairs)], data_label='Neutral', offset=-0.125,
                           jitter=0.05, color='k', ax=ax[idd])
            erp.scatterbar(data[1][~np.isnan(pairs)], pairs[~np.isnan(pairs)], data_label='Shock', offset=0.125,
                           jitter=0.05, color='r', ax=ax[idd])

            # Plot shuffled CIs
            nshuf = shuf[0].shape[3]
            if nshuf > 0:
                unique_pairs = np.unique(pairs[~np.isnan(pairs)])
                for id, lstyle in enumerate(['k--', 'r--']):  # plot each CI independently for now, will need to take average if things all look the same
                    # More conservative CIs here for legacy purposes
                    # CI = np.asarray([mean_CI(shuf[id].reshape(-1, nshuf)[pairs.reshape(-1) == pair_id])
                    #                         for pair_id in unique_pairs])
                    CI = np.asarray([get_CI(np.nanmean(shuf[id].reshape(-1, nshuf)[pairs.reshape(-1) == pair_id],
                                                       axis=0)) for pair_id in unique_pairs])
                    CIlines = ax[idd].plot(np.matlib.repmat(unique_pairs, 3, 1).transpose(), CI, lstyle)
                    CIlines[1].set_linestyle('-')

            if self.best_rot:
                title_append = ' at optimal rotation'
            elif not self.best_rot:
                title_append= ' no rotation'
            ax[idd].set_title(titles[idd] + ': ' + type + title_append)
            ax[idd].set_xticks(np.unique(pairs[~np.isnan(pairs)]))
            ax[idd].set_xticklabels(labels)
            if idd == 2:
                ax[idd].legend()

        return ax


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
        idc = self.idmat(arena1, arena2)  # pick out correct index for correlation matrix...
        save_flag = False  # Set up saving plots

        ascat, axlines = [], []
        linetypes = ['-', '--', '-.']
        for idg, group in enumerate(groups):
            if idg == (len(groups) - 1) and save_fig is True:
                save_flag = True
            shuf_use = self.data[type]['shuf'][group][idc]
            _, _, atemp, axl, aCI = plot_pfcorr_bygroup(self.data[type]['data'][group][idc], arena1, arena2,
                                                   type + ' Correlations ' + arena1 + ' v ' + arena2,
                                                   best_rot=self.best_rot, shuf_mat=shuf_use,
                                                   ax_use=ax, color=palette[idg], offset=-0.1, save_fig=save_flag,
                                                   group_desig=group_desig, linetype=linetypes[idg])
            ascat.append(atemp)
            axlines.append(axl)
        ax.legend(groups)

        return fig, ax, ascat, axlines, aCI

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
            open_CIs = [get_CI(np.nanmean(shuf_use[group][0].reshape(-1, nshuf)[epoch_mat[idg].reshape(-1) == epoch_num],
                                       axis=0)) for idg, group in enumerate(groups)]
            shock_CIs = [get_CI(np.nanmean(shuf_use[group][1].reshape(-1, nshuf)[epoch_mat[idg].reshape(-1) == epoch_num],
                                        axis=0)) for idg, group in enumerate(groups)]

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
    # ssoe = SessionStability(plot_type='odd/even')
    pfh = PlaceFieldHalf('Marble07', 'Open', -2, plot_type='half', quickload=True)
    # pfh.calc_half_corrs()
    pass

