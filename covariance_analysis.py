import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from os import path
import sys, os
import scipy.io as sio
from pickle import dump, load
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import freezing_analysis as fa
import session_directory as sd
import helpers
from helpers import contiguous_regions
import eraser_reference as err


class CovMat:
    """Calculate covariance matrix for a given session
    :param bin_size (seconds): size to bin PSAbool for calculating covariance
    :param exclude_events: str, typically "freeze_onset"
    :param exclude_buffer: tuple len = 2, typically (2, 2) for #seconds before/after event to exclude """
    def __init__(self, mouse: str, arena: str, day: int, bin_size: float = 0.5,
                 max_event_num: int or None = None, exclude_events=None, exclude_buffer=None,
                 keep_events=None, keep_buffer=None):

        if exclude_events is not None:
            assert keep_events is None, '"keep_events" and "exclude_events" cannot both be specified. One must be None.'
        elif keep_events is not None:
            assert exclude_events is None, \
                '"keep_events" and "exclude_events" cannot both be specified. One must be None.'

        # Save session info
        self.mouse = mouse
        self.arena = arena
        self.day = day
        self.exclude_events = exclude_events
        self.exclude_buffer = exclude_buffer
        self.keep_events = keep_events
        self.keep_buffer = keep_buffer
        self.max_event_num = max_event_num

        # ID working directory
        dir_use = pf.get_dir(mouse, arena, day)
        self.dir_use = Path(dir_use)

        # Load in relevant data
        try:
            self.PF = pf.load_pf(mouse, arena, day)
            _, self.freeze_bool = fa.get_freeze_bool(mouse, arena, day)
            self.freeze_ind = np.where(self.freeze_bool)[0]
            md = fa.MotionTuning(mouse, arena, day)
            self.freeze_starts = md.select_events('freeze_onset')
            self.freeze_ends = md.select_events('move_onset')

            # Fix previously calculated occmap
            self.PF.occmap = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap)
            self.PSAbool = self.PF.PSAbool_align
            if exclude_events is not None and keep_events is None:  # exclude some peri-event times from consideration
                if exclude_events == 'freeze_onset':
                    if max_event_num is None:  # exclude all peri-freezing times
                        include_bool = np.bitwise_not(md.get_peri_event_bool(self.exclude_events, self.exclude_buffer))
                    else:
                        nevents_exclude = np.max([len(self.freeze_starts) - max_event_num, 0])
                        # Print out # events to exclude as a sanity check!
                        # print(f'{nevents_exclude} events out of {len(self.freeze_starts)} excluded for day {day}')
                        include_bool = np.bitwise_not(md.get_peri_event_bool(self.exclude_events, self.exclude_buffer,
                                                      nevents_max=nevents_exclude))
                self.PSAbool = self.PSAbool[:, include_bool]

            # Keep only certain events such as peri-freeze times. Lots of double negatives in code, there
            # is probably a better way
            if keep_events is not None and exclude_events is None:
                assert keep_events == 'freeze_onset'
                if max_event_num is None:  # exclude all peri-freezing times
                    include_bool = np.bitwise_not(md.get_peri_event_bool(self.keep_events, self.keep_buffer,
                                                                         apply_not=True))
                else:
                    include_bool = np.bitwise_not(md.get_peri_event_bool(self.keep_events, self.keep_buffer,
                                                                         apply_not=True, nevents_max=max_event_num))
                self.PSAbool = self.PSAbool[:, include_bool]

            self.SR = self.PF.sr_image

        except FileNotFoundError:
            print(f'No position data found for {mouse} {arena} day {day}, loading neural data only')
            self.PF = None
            self.freeze_bool, self.freeze_ind = None, None
            self.freeze_starts, self.freeze_ends = None, None
            neuraldata = sio.loadmat(path.join(self.dir_use, 'FinalOutput.mat'))
            self.PSAbool = neuraldata['PSAbool']
            self.SR = neuraldata['SampleRate'].squeeze()

        # First bin events
        PSAsmooth, PSAbin = [], []
        self.bin_size = bin_size
        for psa in self.PSAbool:
            PSAbin.append(fa.bin_array(psa, int(bin_size * self.SR)))  # Create non-overlapping bin array
        self.PSAbin = np.asarray(PSAbin)
        self.PSAbinz = stats.zscore(PSAbin, axis=1)  # Seems to give same results

        # Now calculate covariance matrix for all your cells using binned array
        self.cov_mat = np.cov(self.PSAbin)
        self.cov_matz = np.cov(self.PSAbinz)


class CovMatReg:
    """Track covariance matrix across days, playing base day in lower diagonal, reg day in upper diagonal and zeroing
    out the diagonal
    :param restrict_num_baseline: bool, False (default) do NOT limit # of events used for calculating
    covariance on days -2 and -1
    :param **kwargs: used to feed 'exclude_events' and 'exclude_buffer' into CovMat class"""
    def __init__(self, mouse: str, base_arena: str, base_day: str, reg_arena: str, reg_day: str,
                 bin_size: float = 0.5, max_event_num: int or None = None, exclude_events=None,
                 exclude_buffer=(2, 2), keep_events=None, keep_buffer=(2, 2)):

        self.mouse = mouse
        self.base_arena = base_arena
        self.base_day = base_day
        self.reg_arena = reg_arena
        self.reg_day = reg_day

        # don't downsample # events to match baseline day if already looking at baseline day
        # Set max events to 1000, much more than we will ever see.
        max_event_base = 1000 if base_day in [-2, -1] else max_event_num
        max_event_reg = 1000 if reg_day in [-2, -1] else max_event_num

        # Calculate covariance matrices for each day
        self.CovMatbase = CovMat(mouse, base_arena, base_day, bin_size, max_event_num=max_event_base,
                                 exclude_events=exclude_events, exclude_buffer=exclude_buffer,
                                 keep_events=keep_events, keep_buffer=keep_buffer)
        self.CovMatreg = CovMat(mouse, reg_arena, reg_day, bin_size, max_event_num=max_event_reg,
                                exclude_events=exclude_events, exclude_buffer=exclude_buffer,
                                keep_events=keep_events, keep_buffer=keep_buffer)

        # Now register across days
        neuron_map = pfs.get_neuronmap(mouse, base_arena, base_day, reg_arena, reg_day, batch_map_use=True)
        self.neuron_map = neuron_map
        self.nneurons_base = neuron_map.shape[0]

        self.mat_type = None
        self.base_neurons = None
        self.covmatreg = None
        self.covmatneurons = None
        self.sigcovpairs = None

    def cov_across_days(self, neurons: str in ['freeze_onset', 'move_onset', 'all'] or np.ndarray = 'freeze_onset',
                        keep_silent: bool = False, buffer_sec=(6, 6), overwrite: bool = True):
        """Track covariance matrix across days - puts base day in lower diagonal, reg day in upper diagonal, and
        zeros out diagonal"""

        assert (isinstance(neurons, str) and neurons in ['freeze_onset', 'move_onset', 'all']) or isinstance(neurons,
                                                                                                             np.ndarray)
        if isinstance(neurons, str) and neurons in ['freeze_onset', 'move_onset']:
            MDbase = fa.MotionTuning(self.mouse, self.base_arena, self.base_day, buffer_sec=buffer_sec)
            sig_neurons = MDbase.get_sig_neurons(events=neurons, buffer_sec=buffer_sec)
            self.mat_type = neurons
        elif isinstance(neurons, str) and neurons == 'all':
            sig_neurons = np.arange(self.nneurons_base)
            self.mat_type = neurons
        else:
            sig_neurons = neurons
            self.mat_type = 'custom'
            self.base_neurons = neurons

        sig_neurons_reg = self.neuron_map[sig_neurons]
        sigbool = sig_neurons_reg > -1
        sig_neurons_reg = sig_neurons_reg[sigbool]
        sig_neurons_base = sig_neurons[sigbool]

        if not keep_silent:
            covz_base = self.CovMatbase.cov_matz[sig_neurons_base][:, sig_neurons_base]
            covz_reg = self.CovMatreg.cov_matz[sig_neurons_reg][:, sig_neurons_reg]
        elif keep_silent:
            covz_base = self.CovMatbase.cov_matz[sig_neurons][:, sig_neurons]
            covz_reg = np.zeros_like(covz_base)
            covz_reg[np.outer(sigbool, sigbool)] = self.CovMatreg.cov_matz[sig_neurons_reg][:, sig_neurons_reg].reshape(-1)

        covz_comb = np.tril(covz_base, -1) + np.triu(covz_reg, 1)

        if overwrite:
            self.covmatreg = covz_comb
            self.covmatneurons = neurons  # track which neurons are in covmatreg

        return covz_comb

    def sig_cov_across_days(self, thresh: float = 2, keep_silent: bool = False):
        """Track any neurons with z-scored covariance above thresh across to the next day. Returns a 2xn array of cell-pair
        covariances, row0 = base day cov, row1 = reg day cov"""

        # Register covariance for all cells across days
        covmatreg = self.cov_across_days('all', keep_silent=keep_silent, overwrite=False)

        # Grab lower (base) pairs and upper (reg) pairs
        ibase, jbase = np.tril_indices_from(covmatreg, -1)
        covbase = covmatreg[ibase, jbase]
        covreg = covmatreg.T[ibase, jbase]

        # z score and id pairs above thresh
        base_std = np.std(covbase)
        base_mean = np.mean(covbase)
        covbasez = (covbase - base_mean)/base_std
        sigbool = covbasez > thresh

        # Grab only significant pairs
        sigcovbase = covbase[sigbool]
        sigcovreg = covreg[sigbool]

        return np.vstack((sigcovbase, sigcovreg))


class PairwiseCoactivation:
    """Class to calculate pairwise activation between neurons and calculate significance based on trial shuffling"""
    def __init__(self, mouse, arena, day, buffer_sec, event_type='freeze_onset'):
        self.session = {'mouse': mouse, 'arena': arena, 'day': day}
        self.buffer_sec = buffer_sec
        self.event_type = event_type

        self.MD = fa.MotionTuning(mouse, arena, day, buffer_sec=buffer_sec)
        self.pval = None

    def calc_pw_significance(self, alpha=0.05, nbins = 3, nshifts=1000,
                             neurons: str in ['all', 'freeze_onset']='freeze_onset'):
        """Calculate pairwise significance by shifting trials to get chance level.
        :param alpha: significance threshold (1 - proportion of shuffles exceeding actual covariance)
        :param nbins: must have more than this number of consecutive bins with p < alpha to be considered significant
        :param nshifts: int, number of shifts to perform"""

        # First select neurons to use
        assert neurons in ['all', 'freeze_onset']
        cells_to_use = 'all'
        if neurons == 'freeze_onset':
            cells_to_use = self.MD.get_sig_neurons(events=self.event_type, buffer_sec=self.buffer_sec)

        # Calculate pairwise activity
        pw_co_all, pw_co_all_prob, times = self.MD.calc_pw_coactivity('freeze_onset', buffer_sec=self.buffer_sec,
                                                                      cells_to_use=cells_to_use)

        # Calculate shuffled activity
        pw_co_all_shift, pw_co_all_prob_shift, times = self.MD.calc_pw_coactivity('freeze_onset',
                                                                                  buffer_sec=self.buffer_sec,
                                                                                  trial_shift=nshifts,
                                                                                  cells_to_use=cells_to_use)

        # Calculate diff b/w actual and shuffled activity - gives npairs x nshifts x ntime_bins array
        pwdiff = (pw_co_all_prob[:, :, None] - pw_co_all_prob_shift).swapaxes(1, 2)

        # Calculate p-value
        pval = 1 - (pwdiff > 0).sum(axis=1)/pwdiff.shape[1]

        # Calculate significance
        sig_bool = np.asarray([(np.diff(contiguous_regions(p < alpha), axis=1) > nbins).any() for p in pval],
                              dtype=bool)

        return pval, sig_bool


def group_cov_across_days(bin_size: float, arena1: str in ['Open', 'Shock'], arena2: str in ['Open', 'Shock'],
                          neurons: str in ['freeze_onset', 'move_onset', 'all'] or np.ndarray = 'freeze_onset',
                          keep_silent: bool = False, buffer_sec: int or list or tuple = (6, 6),
                          base_days: list = [-2, -1, 4, 1, 2, 4], reg_days: list = [-1, 4, 1, 2, 7, 2],
                          match_event_num: bool = False, exclude_events=None, exclude_buffer=(6, 6),
                          keep_events=None, keep_buffer=(6, 6)):
    """Assemble all across-day covariance matrices into a dictionary for easy manipulating later on
    :param match_event_num: set to True to ensure that any covariance calculated from days after -2/-1 using motion
    tuned cells uses the same # of events on average as from day -2/-1.
    :param **kwargs: used to exclude peri-event times from being analyzed with "exclude_events" and "exclude_buffer"
    params."""

    group_plot = [err.learners, err.nonlearners, err.ani_mice_good]
    group_names = ['Learners', 'Non-learners', 'ANI']

    cov_dict = dict.fromkeys(group_names)
    for group, name in zip(group_plot, group_names):
        cov_dict[name] = dict.fromkeys(group)
        for mouse in group:
            # day1 = [-2, -1, 4, 1, 2, 4]
            # day2 = [-1, 4, 1, 2, 7, 2]
            cov_dict[name][mouse] = {}
            if match_event_num:  # Randomly downsample # freezing events to match that of day -2/-1 mean
                # assert neurons != 'all'
                events_use = exclude_events if exclude_events is not None else keep_events
                nevents_baseline = [len(fa.MotionTuning(mouse, arena1, -2).select_events(events_use)),
                                    len(fa.MotionTuning(mouse, arena1, -1).select_events(events_use))]
                nevents_max = np.mean(nevents_baseline).astype(int)
            for ida, (d1, d2) in tqdm(enumerate(zip(base_days, reg_days)), desc=mouse):
                cov_dict[name][mouse][f'{d1}_{d2}'] = []
                try:
                    # blockPrint()
                    if not match_event_num:
                        CMR = CovMatReg(mouse, arena1, d1, arena2, d2, bin_size=bin_size,
                                        exclude_events=exclude_events, exclude_buffer=exclude_buffer,
                                        keep_events=keep_events, keep_buffer=keep_buffer)
                    else:
                        CMR = CovMatReg(mouse, arena1, d1, arena2, d2, bin_size=bin_size,
                                        exclude_events=exclude_events, max_event_num=nevents_max,
                                        exclude_buffer=exclude_buffer,
                                        keep_events=keep_events, keep_buffer=keep_buffer)
                    covz_comb = CMR.cov_across_days(neurons, keep_silent=keep_silent, buffer_sec=buffer_sec)
                    # enablePrint()
                    cov_dict[name][mouse][f'{d1}_{d2}'] = covz_comb
                except FileNotFoundError:
                    print(f'{mouse} {arena1} day {d1} to {arena2} {d2} session(s) missing')

    return cov_dict


def group_sig_cov_across_days(bin_size: float, arena1: str in ['Open', 'Shock'], arena2: str in ['Open', 'Shock'],
                              base_days: list = [4, 4, 4, 4], reg_days: list = [-2, -1, 1, 2],
                              thresh: float = 2, keep_silent: bool = False):
    """Assemble all across-day covariance matrices for cell-pairs that have covariance above thresh on base day"""

    group_plot = [err.learners, err.nonlearners, err.ani_mice_good]
    group_names = ['Learners', 'Non-learners', 'ANI']

    cov_dict = dict.fromkeys(group_names)
    for group, name in zip(group_plot, group_names):
        cov_dict[name] = dict.fromkeys(group)
        for mouse in group:
            cov_dict[name][mouse] = {}
            for ida, (d1, d2) in tqdm(enumerate(zip(base_days, reg_days)), desc=mouse):
                cov_dict[name][mouse][f'{d1}_{d2}'] = []
                try:
                    # blockPrint()
                    CMR = CovMatReg(mouse, arena1, d1, arena2, d2, bin_size=bin_size)
                    covz_comb = CMR.sig_cov_across_days(thresh, keep_silent=keep_silent)
                    # enablePrint()
                    cov_dict[name][mouse][f'{d1}_{d2}'] = covz_comb
                except FileNotFoundError:
                    print(f'{mouse} {arena1} day {d1} to {arena2} {d2} session(s) missing')

    return cov_dict


def plot_pw_cov_across_days(dict_use, ndays, include_silent, **kwargs):
    """Plots pairwise covariance across days"""
    for group_name in dict_use.keys():
        group_dict = dict_use[group_name]
        for mouse_name in group_dict.keys():
            mouse_dict = group_dict[mouse_name]
        #         ndays = len(mouse_dict.keys())
            fig, ax = plt.subplots(1, ndays, figsize=(3*ndays, 2.5))
            for idd, d1_d2 in enumerate(mouse_dict.keys()):
                cov_mat = mouse_dict[d1_d2]
                day1, day2 = d1_d2.split('_')
                fa.scatter_cov_across_days(cov_mat, include_silent=include_silent, xlabel=f'Day {day1}',
                                           ylabel=f'Day {day2}', ax=ax[idd], **kwargs)
            fig.suptitle(mouse_name)


def get_cov_pairs_across_days(dict_use, include_silent):
    pairs_dict = dict.fromkeys(dict_use.keys())
    for group_name in dict_use.keys():
        group_dict = dict_use[group_name]
        pairs_dict[group_name] = dict.fromkeys(group_dict.keys())
        for mouse_name in group_dict.keys():
            mouse_dict = group_dict[mouse_name]
            pairs_dict[group_name][mouse_name] = dict.fromkeys(mouse_dict.keys())
            for idd, d1_d2 in enumerate(mouse_dict.keys()):
                cov_mat = mouse_dict[d1_d2]
                cov_pairs = fa.scatter_cov_across_days(cov_mat, include_silent=include_silent, plot=False)
                pairs_dict[group_name][mouse_name][d1_d2] = cov_pairs

    return pairs_dict


def cov_dict_to_df(dict_use, baseline_dict_use, register: bool = False, include_silent: bool = True,
                   group_ctrls=True):
    """Calculate z-scored covariance of all cells in dict compared to pre-shock days for all mice
     and put into DataFrame. Also designates animals into Ctrl or ANI groups.
     :param dict_use: dictionary containing covariance mats of cells you want to look at, e.g. freeze cells
     :param baseline_dict_use: dictionary containing covariance mats of cells you want to normalize by, typically all cells
     :param register: bool, False (default) = just use base day covariance, True = use reg day (for tracking across days).
     :param include_silent: bool
     :param group_ctrls: bool, group Learners and Non-Learners into a control group"""

    day_code, group_code, sigzmean, sigzall = [], [], [], []  # pre-allocate for plotting!
    for group_name in dict_use.keys():
        group_dict = dict_use[group_name]

        for mouse_name in group_dict.keys():

            # Grab covariance of cells during baseline (day -2 and -1) sessions and calculate mean and std.
            cov_baseline = []
            for d1_d2 in ['-2_-1', '-1_4']:
                base_mat_use = baseline_dict_use[group_name][mouse_name][d1_d2]
                base_cov, _ = get_cov_pairs_from_mat(base_mat_use, None, include_silent=include_silent)
                cov_baseline.extend(base_cov)
            mean_baseline = np.mean(cov_baseline)
            std_baseline = np.std(cov_baseline)

            # Grab covariance for cells for each session and normalize by baseline session
            mouse_dict = dict_use[group_name][mouse_name]
            for d1_d2 in mouse_dict.keys():
                try:
                    mat_use = mouse_dict[d1_d2]
                    base_cov, reg_cov = get_cov_pairs_from_mat(mat_use, None, include_silent=include_silent)
                    cov_use = reg_cov if register else base_cov  # grab appropriate covariances to use.
                    sigzall.append((cov_use - mean_baseline) / std_baseline)
                    sigzmean.append(np.nanmean(sigzall[-1]))
                    if group_ctrls:
                        group_code.append('ANI') if group_name == 'ANI' else group_code.append('Ctrl')
                    else:
                        group_code.append(group_name)
                    day_code.append(d1_d2)
                except KeyError:
                    pass

    sigz_df = pd.DataFrame(data={'d1_d2': day_code, 'Group': group_code, 'cov_z_mean': sigzmean})
    return sigz_df


def get_cov_pairs_from_array(cov_array: np.ndarray, include_silent: bool = False):
    """Grab pairs of cells to plot versus one-another across sessions from a pairwise covariance array
        (base day first row, reg day second row)"""

    assert isinstance(include_silent, bool)
    assert cov_array.shape[0] == 2

    if not include_silent:
        silent_bool = cov_array[1] == 0
        active_bool = np.bitwise_not(silent_bool)
        array_use = cov_array[:, active_bool]
    else:
        array_use = cov_array

    base_cov = array_use[0]
    reg_cov = array_use[1]

    return base_cov, reg_cov


def get_cov_pairs_from_mat(cov_mat: np.ndarray, cells: np.ndarray or None,
                           include_silent: bool = False):
    """Grab pairs of cells to plot versus one-another across sessions from a pairwise covariance matrix
    (base day below diagonal, reg day above diagonal)"""
    if cov_mat.shape[0] == cov_mat.shape[1]:  # Run on matrix if input is square array
        if cells is not None:
            cov_mat = cov_mat[cells][:, cells]

        assert isinstance(include_silent, bool)
        if not include_silent:
            silent_bool = np.triu(cov_mat).sum(axis=1) == 0
            active_bool = np.bitwise_not(silent_bool)
            active_outer = np.outer(active_bool, active_bool)
            nactive = active_bool.sum()
            mat_use = cov_mat[active_outer].reshape((nactive, nactive))
        else:
            mat_use = cov_mat

        # Curate and grab base vs reg covariance pairs
        l_inds = np.tril_indices_from(mat_use, -1)
        base_cov = mat_use[l_inds]
        reg_cov = mat_use.T[l_inds]

    elif cov_mat.shape[0] == 2 and cov_mat.shape[0] < cov_mat.shape[1]:  # Run on array if not a square matrix
        if cells is not None:
            print('2xn covariance array provided is not compatible with any input to "cells" parameter. Exiting.')
            return None, None
        base_cov, reg_cov = get_cov_pairs_from_array(cov_mat, include_silent=include_silent)

    return base_cov, reg_cov


def get_group_PBE_rasters(animal_list, group_name, buffer_sec=(6, 6), event_type='freeze_onset', sr_match=20):
    """Gets rasters of population level calcium activity centered on specified events default = freeze onset"""

    # Set up times for all PBE rasters
    times = np.arange(-buffer_sec[0], buffer_sec[1], 1 / sr_match)

    nanimals = len(animal_list)
    PBEdict = {}
    for day in [-2, -1, 4, 1, 2]:
        PBErast_comb, PBErast_combz, times_comb = [], [], []
        for animal in animal_list:
            MD1 = fa.MotionTuning(animal, 'Shock', day)
            PBErast = fa.get_PE_raster(MD1.PSAbool.sum(axis=0), MD1.select_events(event_type),
                                       sr_image=MD1.sr_image, buffer_sec=buffer_sec)
            # Sum up cells active and divide by total to get proportion active before each event
            PBErast_prop = PBErast.mean(axis=0) / MD1.PSAbool.shape[0]

            # z-score proportions
            prop_active = MD1.PSAbool.sum(axis=0) / MD1.PSAbool.shape[0]
            PBErast_propz = (PBErast_prop - prop_active.mean()) / prop_active.std()

            if MD1.sr_image != sr_match:  # interpolate values if sample rate doesn't match to make data compatible
                rast_mean_interp = []
                times_sr = np.arange(-buffer_sec[0], buffer_sec[1], 1 / MD1.sr_image)
                PBErast_comb.extend(np.interp(times, times_sr, PBErast_prop))
                PBErast_combz.extend(np.interp(times, times_sr, PBErast_propz))
            else:
                PBErast_comb.extend(PBErast_prop)
                PBErast_combz.extend(PBErast_propz)
            times_comb.extend(times)  # aggregate times

        # Assemble into dataframes for easy plotting later on
        PBEdict[day] = pd.DataFrame({'times': np.array(times_comb).reshape(-1),
                                     'act_neuron_ratio': np.array(PBErast_comb).reshape(-1),
                                     'act_neuron_ratio_z': np.array(PBErast_combz).reshape(-1),
                                     'group': [group_name] * len(np.array(times_comb).reshape(-1)),
                                     'day': [day] * len(np.array(times_comb).reshape(-1))})

    return PBEdict


def gen_pw_coact(event_type, arena='Shock', buffer_sec=(6, 6), buffer_sec_filt=(6, 6), sr_match=20,
                 days=[-2, -1, 4, 1, 2], cell_filt: str in ['all', 'freeze_cells'] = 'all', **kwargs):
    """Get pairwise coactivity for all animals and fold it nicely into a dataframe for easy plotting.  'pw_co' is the
    mean # of coactivation events for all pairs. 'pw_co_prob' is the mean probability that each pair of cells is
    coactive on a given frame, e.g.  #frames coactive / # events.
    'buffer_sec' gives you the extent of the axes while 'buffer_sec_filt' is used to select freeze cells.  For example
    if you want to only consider cells with significant freeze tuning between -2 and 2 seconds from freeze onset, but
    you want your plot to extend out to +/- 6 seconds, you would use buffer_sec_filt=(2, 2) and buffer_sec=(6, 6).

    **kwargs goes to MotionTuning.get_sig_neurons(...)"""
    # Set up times for all PBE rasters
    times = np.arange(-buffer_sec[0], buffer_sec[1], 1 / sr_match)

    coact_df_list = []
    for animal_list, grp_name in zip((err.learners, err.nonlearners, err.ani_mice_good),
                                     ('Learners', 'Non-learners', 'ANI')):
        pw_co_all, pw_co_prob_all, times_all, day_all, grp_all, animal_all = [], [], [], [], [], []
        for idm, mouse in enumerate(animal_list):
            for day in days:

                # Get motion tuning curves
                MD1 = fa.MotionTuning(mouse, arena, day, buffer_sec=buffer_sec_filt)
                MD1.gen_pe_rasters(buffer_sec=buffer_sec)

                # Grab appropriate cells
                if cell_filt == 'all':
                    cells_to_use = 'all'
                else:
                    cells_to_use = MD1.get_sig_neurons(event_type, buffer_sec=buffer_sec_filt)

                # Calculate coactivation
                pwco, pwcoprob, times = MD1.calc_pw_coactivity(events=event_type, buffer_sec=buffer_sec,
                                                               cells_to_use=cells_to_use)

                # Append everything into a long list
                if pwco is not None:
                    pw_co_all.extend(pwco.mean(axis=0))
                    pw_co_prob_all.extend(pwcoprob.mean(axis=0))
                    times_all.extend(times)
                    day_all.extend([day] * len(times))
                    grp_all.extend([grp_name] * len(times))
                    animal_all.extend([idm] * len(times))

        # Make your dataframe
        coact_df = pd.DataFrame({'time': times_all, 'pw_co': pw_co_all, 'pw_co_prob': pw_co_prob_all,
                                 'day': day_all, 'group': grp_all, 'mouse': animal_all})
        coact_df_list.append(coact_df)

    # Concat to create final dataframe
    coact_df_all = pd.concat(coact_df_list)

    return coact_df_all


def blockPrint():
    # Helper functions to block printing output
    # Disable
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    # Helper functions to block printing output
    # Enable
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    freeze_silent_cov = group_cov_across_days(bin_size=0.5, arena1='Shock', arena2='Shock',
                                              neurons='freeze_onset', keep_silent=True, buffer_sec=(4, 4))

