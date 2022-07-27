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
    """Calculate covariance matrix for a given session"""
    def __init__(self, mouse: str, arena: str, day: int, bin_size: float = 0.5):
        # Save session info
        self.mouse = mouse
        self.arena = arena
        self.day = day

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
    out the diagonal"""
    def __init__(self, mouse: str, base_arena: str, base_day: str, reg_arena: str, reg_day: str,
               bin_size: float = 0.5):

        self.mouse = mouse
        self.base_arena = base_arena
        self.base_day = base_day
        self.reg_arena = reg_arena
        self.reg_day = reg_day

        self.CovMatbase = CovMat(mouse, base_arena, base_day, bin_size)
        self.CovMatreg = CovMat(mouse, reg_arena, reg_day, bin_size)

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
            MDbase = fa.MotionTuning(self.mouse, self.base_arena, self.base_day)
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


def group_cov_across_days(bin_size: float, arena1: str in ['Open', 'Shock'], arena2: str in ['Open', 'Shock'],
                          neurons: str in ['freeze_onset', 'move_onset', 'all'] or np.ndarray = 'freeze_onset',
                          keep_silent: bool = False, buffer_sec: int or list or tuple = (6, 6),
                          base_days: list = [-2, -1, 4, 1, 2, 4], reg_days: list = [-1, 4, 1, 2, 7, 2]):
    """Assemble all across-day covariance matrices into a dictionary for easy manipulating later on"""

    group_plot = [err.learners, err.nonlearners, err.ani_mice_good]
    group_names = ['Learners', 'Non-learners', 'ANI']

    cov_dict = dict.fromkeys(group_names)
    for group, name in zip(group_plot, group_names):
        cov_dict[name] = dict.fromkeys(group)
        for mouse in group:
            # day1 = [-2, -1, 4, 1, 2, 4]
            # day2 = [-1, 4, 1, 2, 7, 2]
            cov_dict[name][mouse] = {}
            for ida, (d1, d2) in tqdm(enumerate(zip(base_days, reg_days)), desc=mouse):
                cov_dict[name][mouse][f'{d1}_{d2}'] = []
                try:
                    # blockPrint()
                    CMR = CovMatReg(mouse, arena1, d1, arena2, d2, bin_size=bin_size)
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
                    sigzmean.append(np.mean(sigzall[-1]))
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


def blockPrint():
    # Helper functions to block printing output
    # Disable
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    # Helper functions to block printing output
    # Enable
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    group_cov_across_days(0.5, 'Shock', 'Shock')