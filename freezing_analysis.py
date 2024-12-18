import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from os import path
import scipy.io as sio
from pickle import dump, load
from pathlib import Path
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA as skFastICA
from sklearn.decomposition import PCA as skPCA
import copy

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers
from helpers import contiguous_regions
import eraser_reference as err

class Raster:
    def __init__(self, raster_array, buffer_sec=(2, 2), SR=20):
        self.raster = raster_array
        self.times = np.linspace(-buffer_sec[0], buffer_sec[1], np.sum(buffer_sec)*SR+1)

    @property
    def raster_mean(self):
        raster_mean = (
            np.nanmean(self.raster, axis=0) if self.raster is not None else None
        )
        return raster_mean

    def get_mean_peak(self):
        """Get peak of mean raster and index where it occurs"""
        idx = np.argmax(self.raster_mean)

        return idx, self.raster_mean[idx]

    def get_mean_trough(self):
        """Get trough of mean raster and index where it occurs"""
        idx = np.argmin(self.raster_mean)

        return idx, self.raster_mean[idx]


class RasterGroup:
    def __init__(self, raster_array_group, buffer_sec=(2, 2), SR=20):
        self.Raster = []
        for raster in raster_array_group:
            self.Raster.append(
                Raster(
                    raster,
                    buffer_sec=buffer_sec,
                    SR=SR,
                )
            )

    def sort_rasters(
        self,
        sortby: list or np.ndarray or str = "peak_time",
        norm_each_row: None or str in ["max", "z"] = "max",
    ):

        assert isinstance(sortby, (list, np.ndarray)) or (
            isinstance(sortby, str) and sortby in ["peak_time", "trough_time"]
        )
        if isinstance(sortby, str) and (
            sortby == "peak_time" or sortby == "trough_time"
        ):
            peak_idx = []
            for rast in self.Raster:
                pid, _ = (
                    rast.get_mean_peak()
                    if sortby == "peak_time"
                    else rast.get_mean_trough()
                )
                peak_idx.append(pid)
            sort_ids = np.argsort(peak_idx)
        else:
            sort_ids = sortby

        sorted_mean_rast = np.array([self.Raster[idx].raster_mean for idx in sort_ids])

        # Normalize each row to itself
        if norm_each_row == "max":
            sorted_mean_rast = sorted_mean_rast / sorted_mean_rast.max(axis=1)[:, None]
        elif norm_each_row == "z":
            sorted_mean = np.array(
                [np.nanmean(self.Raster[idx].raster.reshape(-1)) for idx in sort_ids]
            )
            sorted_std = np.array(
                [np.nanstd(self.Raster[idx].raster.reshape(-1)) for idx in sort_ids]
            )
            sorted_mean_rast = (sorted_mean_rast - sorted_mean[:, None]) / sorted_std[
                :, None
            ]
        return sorted_mean_rast, sort_ids


class MotionTuning:
    """Identify and plot freeze and motion related cells
    **kwargs: inputs for getting freezing in eraser_plot_functions.get_freezing.
    """
    def __init__(self, mouse, arena, day, buffer_sec=(4, 4), **kwargs):
        self.session = {'mouse': mouse, 'arena': arena, 'day': day}
        self.buffer_sec = buffer_sec

        # ID working directory
        dir_use = pf.get_dir(mouse, arena, day)
        self.dir_use = Path(dir_use)

        # Get PSAbool and freezing info
        self.sr_image = pf.get_im_sample_rate(mouse, arena, day)
        self.PSAbool, self.freeze_bool = get_freeze_bool(mouse, arena, day, **kwargs)
        self.event_rates = self.PSAbool.sum(axis=1)/self.PSAbool.shape[1] * self.sr_image
        self.freezing_indices, self.freezing_times = get_freezing_times(mouse, arena, day, zero_start=True)

        # Get initial estimate of motion-modulated vs freeze modulated cells - very rough
        # don't really use - calculate later through function if needs be!
        # print('calculating rough modulation')
        # self.p, self.ER = calc_sig_modulation(mouse, arena, day)
        # self.freeze_cells_rough = np.where(self.p['MMI'] > 0.95)[0]
        # self.move_cells_rough = np.where(self.p['MMI'] < 0.05)[0]

        # Get sample rate
        # im_data_file = path.join(dir_use, 'FinalOutput.mat')
        # im_data = sio.loadmat(im_data_file)
        # try:
        #     self.sr_image = im_data['SampleRate'].squeeze()
        # except KeyError:
        #     self.sr_image = 20

        self.pe_rasters = {'freeze_onset': None, 'move_onset': None}
        self.perm_rasters = {'freeze_onset': None, 'move_onset': None}

        try:  # Load in previously calculated tunings
            self.load_sig_tuning(buffer_sec=buffer_sec)
        except FileNotFoundError:  # if not saved, initialize
            print(f'No tunings found for {mouse} {arena} day {day}: run .get_tuning_sig() and .save_sig_tuning()')
            self.sig = {'freeze_onset': {}, 'move_onset': {}}

    def get_prop_tuned(self, events: str = 'freeze_onset', buffer_sec: tuple = (4, 4), **kwargs):
        """
        Gets proportion of neurons that exhibit freeze or motion related tuning
        :param events: str, 'freeze_onset' (default) or 'move_onset'
        :param kwargs: inputs to get_sig_neurons() to classify freeze or motion related movement
        :param buffer_sec: seconds before/after to consider
        :return:
        """
        ntuned = self.get_sig_neurons(events=events, buffer_sec=buffer_sec, **kwargs).shape[0]
        ntotal = self.sig[events]['pval'].shape[0]

        return ntuned/ntotal

    def get_peri_event_bool(self, events: str = 'freeze_onset', buffer_sec=(4, 4), nevents_max=None, apply_not=False):
        """Generates a boolean identifying +/- buffer_sec from event. Grabs only a random subset of
        events if nevents_max is set and less than total # events found.
        apply_not: apply a bitwise not to the output, e.g. events='freeze_onset', apply=True will give you
        a boolean of all non peri-freeze times. Default = True"""
        events = self.select_events(events)
        event_inds = (events * self.sr_image).astype(int)
        peri_event_bool = np.zeros_like(self.PSAbool[0])

        # Grab only a subset of events
        if nevents_max is not None:
            nevents = len(event_inds)
            if nevents_max < nevents:
                events_subset = np.random.permutation(nevents)[0:nevents_max]
                event_inds = event_inds[events_subset]

        for event_ind in event_inds:
            peri_event_bool[np.max([0, event_ind - buffer_sec[0] * self.sr_image]): np.min(
                [len(peri_event_bool), event_ind + buffer_sec[1] * self.sr_image])] = 1

        if apply_not is False:
            return peri_event_bool.astype(bool)
        else:
            return np.bitwise_not(peri_event_bool.astype(bool))

    def gen_pe_rasters(self, events='freeze_onset', buffer_sec=(4, 4), bin_size: float or None = None):
        """Generate the rasters for all cells and dump them into a dictionary"""
        # Get appropriate event times to use
        if events in ['freeze_onset', 'move_onset']:
            event_starts = self.select_events(events)
        if bin_size is None:
            pe_rasters = [get_PE_raster(psa, event_starts, buffer_sec=buffer_sec,
                                        sr_image=self.sr_image) for psa in self.PSAbool]
        else:
            PSAbin = []
            for psa in self.PSAbool:
                PSAbin.append(bin_array(psa, int(bin_size * self.sr_image)))  # Create non-overlapping bin array
            PSAbin = np.asarray(PSAbin)
            pe_rasters = [get_PE_raster(psa, event_starts, buffer_sec=buffer_sec,
                                        sr_image=1/bin_size) for psa in PSAbin]

        pe_rasters = np.asarray(pe_rasters)
        self.pe_rasters[events] = pe_rasters

        return pe_rasters

    def gen_perm_rasters(self, events='freeze_onset', buffer_sec=(4, 4), nperm=1000):
        """Generate shuffled rasters and dump them into a dictionary"""
        # Get appropriate cells and event times to use
        event_starts = self.select_events(events)

        # Loop through each cell and get its chance level raster
        print('generating permuted rasters - may take up to 1 minute')
        perm_rasters = np.asarray([shuffle_raster(psa, event_starts, buffer_sec=buffer_sec,
                                       sr_image=self.sr_image, nperm=nperm) for psa in self.PSAbool]).swapaxes(0, 1)
        self.perm_rasters[events] = perm_rasters

        return perm_rasters

    def get_tuning_sig(self, events='freeze_onset', buffer_sec=(4, 4), nperm=1000):
        """This function will calculate significance values by comparing event-centered tuning curves to
        chance (calculated from circular permutation of neural activity).
        :param events:
        :param buffer_sec:
        :return:
        """

        # Load in previous tuning
        sig_use = self.sig[events]

        calc_tuning = True
        # Check to see if appropriate tuning already run and stored and just use that, otherwise calculate from scratch.
        if 'nperm' in sig_use:
            # Check that rasters are correct size for buffer_sec and nperms are correct
            # For debugging
            # print(f'pval0shape={sig_use["pval"].shape[1]}')
            # print(int(self.sr_image * np.sum(buffer_sec)))
            if (sig_use['nperm'] == nperm) and \
                    (sig_use['pval'].shape[1] == int(self.sr_image * np.sum(buffer_sec))):
                calc_tuning = False
                pval = sig_use['pval']

        if calc_tuning:
            print(f'calculating significant tuning for buffer_sec={buffer_sec} and nperm={str(nperm)}')
            # check if both regular and permuted raster are run already!
            pe_rasters, perm_rasters = self.check_rasters_run(events=events, buffer_sec=buffer_sec,  nperm=nperm)

            # Now calculate tuning curves and get significance!
            pe_tuning = gen_motion_tuning_curve(pe_rasters)
            perm_tuning = np.asarray([gen_motion_tuning_curve(perm_raster) for perm_raster in perm_rasters])
            pval = (pe_tuning < perm_tuning).sum(axis=0) / nperm

            # Store in class
            self.sig[events]['pval'] = pval
            self.sig[events]['nperm'] = nperm

            # Save to disk to save time in future
            self.save_sig_tuning(buffer_sec=buffer_sec)

        return pval

    def get_sig_neurons(self, events='freeze_onset', buffer_sec=(4, 4), nperm=1000,
                        alpha=0.01, nbins=3, active_prop=0.25):
        """Find freezing neurons as those which have sig < alpha for nbins (consecutive) or more AND are active on
        at least active_prop of events."""

        # Load in significance values at each spatial bin and re-run things if not already there
        pval = self.get_tuning_sig(events=events, buffer_sec=buffer_sec, nperm=nperm)

        # Determine if there is significant tuning of < alpha for nbins (consecutive)
        sig_bool = np.asarray([(np.diff(contiguous_regions(p < alpha), axis=1) > nbins).any() for p in pval],
                              dtype=bool)

        # Load in rasters
        pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)
        nevents = pe_rasters.shape[1]

        # Determine if neurons pass the activity threshold
        nevents_active = pe_rasters.any(axis=2).sum(axis=1)
        active_bool = (nevents_active > active_prop * nevents)

        sig_neurons = np.where(np.bitwise_and(sig_bool, active_bool))[0]

        return sig_neurons

    def calc_pw_coactivity(self, events: str in ['freeze_onset', 'move_onset'] = 'freeze_onset', buffer_sec=(4, 4),
                           sr_match: int = 20, cells_to_use: 'all' or list or np.array = 'all',
                           jitter_frames: None or int = None, trial_shift: None or int = None):
        """Calculate pairwise coactivty of all neurons around freeze or motion onset. Returns pairwise activations
        (total # for each pair), pairwise activation probability, and timestamps relative to each event.
        Less fleixble than calc_pw_coactivity_full but has more tools for estimating chance level, so kept in"""

        assert jitter_frames is None or type(jitter_frames) == int
        # Check if peri-event rasters are already loaded in
        if self.pe_rasters[events] is not None:
            # Check if #frames matches buffer_sec input
            (load_rasters := not (np.sum(buffer_sec)*self.sr_image == self.pe_rasters[events].shape[1]))
        else:
            load_rasters = True

        # Load rasters if necessary
        if load_rasters:
            self.gen_pe_rasters(events, buffer_sec)

        # Select cells to use
        if isinstance(cells_to_use, str) and (cells_to_use == 'all'):
            rasters_use = self.pe_rasters[events]
        else:
            rasters_use = self.pe_rasters[events][cells_to_use]
            # print(f'calculating rasters with {rasters_use.shape[0]} cells')  # for debugging

        # Set up variables
        times = np.arange(-buffer_sec[0], buffer_sec[1], 1 / sr_match)
        nevents = rasters_use[0].shape[0]
        pw_co_all = []

        # iterate through and calculate pairwise coactivity for all neuron pairs
        n = 1
        for rast1 in rasters_use[:-1]:
            # Sum up pairwise activity
            if jitter_frames is None:
                pw_co = np.bitwise_and(rast1, rasters_use[n:]).sum(axis=1)
            else:  # rotate all rasters circularly by specified # of frames
                pw_co = np.bitwise_and(rast1, np.roll(rasters_use[n:], jitter_frames)).sum(axis=1)

            if trial_shift is None:
                pw_co = np.bitwise_and(rast1, rasters_use[n:]).sum(axis=1)
            elif trial_shift:  # randomly permute trial/event order for rast1 to compute chance level pairwise coact.
                shift_array = []
                for i in range(trial_shift):
                    shift_array.append(
                        np.bitwise_and(rast1[np.random.permutation(nevents)], rasters_use[n:]).sum(axis=1))

                pw_co = np.dstack(shift_array)

            if self.sr_image == sr_match:
                pw_co_all.append(pw_co)
            else:  # interpolate if sample rate is off
                times_sr = np.arange(-buffer_sec[0], buffer_sec[1], 1 / self.sr_image)
                nbins_sr = len(times_sr)
                npairs = pw_co.shape[0]
                nframes = len(times)

                if trial_shift is None:
                    pw_co_interp = np.concatenate([np.interp(times, times_sr, co_pair) for co_pair in pw_co])\
                        .reshape(npairs, nframes)
                else:
                    nshifts = trial_shift
                    pw_co_rs = pw_co.swapaxes(1, 2).reshape(-1, nbins_sr)
                    pw_co_interp = np.concatenate([np.interp(times, times_sr, co_pair) for co_pair in pw_co_rs]) \
                        .reshape(npairs, nframes, nshifts)
                pw_co_all.append(pw_co_interp)
            n += 1
        if len(pw_co_all) > 0:
            pw_co_all = np.concatenate(pw_co_all)
            pw_co_prob_all = pw_co_all/nevents

            return pw_co_all, pw_co_prob_all, times

        else:
            return None, None, None

    def calc_pw_coactivity_full(self, events: str in ['freeze_onset', 'move_onset'] = 'freeze_onset',
                                buffer_sec=(4, 4), sr_match: int = 20,
                                cells_to_use: 'all' or list or np.array = 'all'):
        """Calculates pairwise activation probability by trial, return an npairs x nevents x ntimebins array of
        pairwise coactivations (1 = coactive, 0 = not). More flexible than calc_pw_coactivity but less ability
        to shuffle/get chance levels"""

        # Check if peri-event rasters are already loaded in
        if self.pe_rasters[events] is not None:
            # Check if #frames matches buffer_sec input
            (load_rasters := not (np.sum(buffer_sec) * self.sr_image == self.pe_rasters[events].shape[1]))
        else:
            load_rasters = True

        # Load rasters if necessary
        if load_rasters:
            self.gen_pe_rasters(events, buffer_sec)

        # Select cells to use
        if isinstance(cells_to_use, str) and (cells_to_use == 'all'):
            rasters_use = self.pe_rasters[events]
        else:
            rasters_use = self.pe_rasters[events][cells_to_use]
            # print(f'calculating rasters with {rasters_use.shape[0]} cells')  # for debugging

        # Set up variables
        times = np.arange(-buffer_sec[0], buffer_sec[1], 1 / sr_match)
        # iterate through and calculate pairwise coactivity for all neuron pairs
        rast_all = []
        for idn, rast in enumerate(rasters_use[:-1]):
            rast_all.append(np.bitwise_and(rast, rasters_use[(idn + 1):]))

        pw_co_all = np.concatenate(rast_all, axis=0)

        if self.sr_image != sr_match:
            times_sr = np.arange(-buffer_sec[0], buffer_sec[1], 1 / self.sr_image)
            npairs, nevents, _ = pw_co_all.shape
            nbins_interp = len(times)
            pw_co_all_interp = (np.concatenate(
                [np.interp(times, times_sr, co_trial) for co_pair in pw_co_all for co_trial in co_pair])
                            .reshape((npairs, nevents, nbins_interp)))
            pw_co_all = pw_co_all_interp
            times = times

        return pw_co_all, times


    def save_sig_tuning(self, buffer_sec):
        """Saves any significant tuned neuron data"""

        # Append buffer_sec before/after to save name if not default (2, 2)
        if buffer_sec[0] == 2 and buffer_sec[1] == 2:
            save_name = "sig_motion_tuning.pkl"
        else:
            save_name = f"sig_motion_tuning{buffer_sec[0]}_{buffer_sec[1]}.pkl"

        with open(self.dir_use / save_name, 'wb') as f:
            dump(self.sig, f)

    def load_sig_tuning(self, buffer_sec):
        """Loads any previously calculated tunings"""

        # Append buffer_sec before/after to save name if not default (2, 2)
        if buffer_sec[0] == 2 and buffer_sec[1] == 2:
            save_name = "sig_motion_tuning.pkl"
        else:
            save_name = f"sig_motion_tuning{buffer_sec[0]}_{buffer_sec[1]}.pkl"

        with open(self.dir_use / save_name, 'rb') as f:
            self.sig = load(f)

    def check_rasters_run(self, events='freeze_onset', buffer_sec=(4, 4),  nperm=1000):
        """ Verifies if you have already created rasters and permuted rasters and checks to make sure they match.

        :param cells:
        :param events:
        :param buffer_sec:
        :param nperm:
        :return:
        """
        # check if both regular and permuted raster are run already!
        pe_rasters = self.pe_rasters[events]
        perm_rasters = self.perm_rasters[events]
        nbins_use = np.sum([int(buffer_sec[0] * self.sr_image), int(buffer_sec[1] * self.sr_image)])
        if isinstance(pe_rasters, np.ndarray) and isinstance(perm_rasters, np.ndarray):
            ncells, nevents, nbins = pe_rasters.shape
            nperm2, ncells2, nevents2, nbins2 = perm_rasters.shape

            # Make sure you are using the same data format!
            assert ncells == ncells2, '# Cells in data and permuted rasters do not match'
            assert nevents == nevents2, '# events in data and permuted rasters do not match'

            # if different buffer_sec used, re-run full rasters
            if nbins != nbins_use:
                pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)

            # if different buffer_sec or nperm used, re-run permuted rasters
            if nbins2 != nbins_use or nperm2 != nperm:
                perm_rasters = self.gen_perm_rasters(events=events, buffer_sec=buffer_sec, nperm=nperm)

        else:
            if not isinstance(pe_rasters, np.ndarray):
                pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)

            if not isinstance(perm_rasters, np.ndarray):
                perm_rasters = self.gen_perm_rasters(events=events, buffer_sec=buffer_sec, nperm=nperm)

        # else:
        #     pe_rasters, perm_rasters = self.check_rasters_run(events=events, buffer_sec=buffer_sec, nperm=nperm)

        return pe_rasters, perm_rasters

    def select_cells(self, cells, buffer_sec=(4, 4), **kwargs):
        """Select different types of cells for plotting

        :param cells: 'freeze_rough', 'move_rough', 'freeze_fine', 'move_fine', or list of cells to use
        'rough' selects cells that are more tuned to ALL freezing (or moving) times, 'fine' zeros in on
        times within buffer_sec of freezing (or moving) onset
        :param buffer_sec: time before/after freeze (or move) onset to consider when calculating 'fine' tuning
        :param kwargs: See .get_sig_tuning for tweaking motion tuning parameters
        :return: list of cell ids to use that meet the criteria
        """

        # Note, rough tunings not calculated by default, but are kept here in case I need them later
        if cells == 'freeze_rough':
            cell_ids = self.freeze_cells_rough
        elif cells == 'move_rough':
            cell_ids = self.move_cells_rough
        elif cells == 'freeze_fine':
            cell_ids = self.get_sig_neurons(events='freeze_onset', buffer_sec=buffer_sec, **kwargs)
        elif cells == 'move_fine':
            cell_ids = self.get_sig_neurons(events='move_onset', buffer_sec=buffer_sec, **kwargs)
        elif isinstance(cells, np.ndarray) or isinstance(cells, list):
            cell_ids = cells

        return cell_ids

    def plot_pe_rasters(self, cells='freeze_fine', events='freeze_onset', buffer_sec=(4, 4), **kwargs):
        """Plot rasters of cells at either movement or freezing onsets.

        :param cells: str to auto-select either cells that have significant tuning to move or freeze onset
        at a fine ('move_fine' or 'freeze_fine' calculated with buffer_sec of motion/freezing onset)
        or rough ('move_rough' or 'freeze_rough', tuning calculated across all freezing/motion timepoints) timescale.
        Can also be a list of other events
        :param events: 'move_onset' or 'freeze_onset'.
        :param buffer_sec: int or size 2, array like of time(s) +/- event to plot
        :param kwargs:
        :return:
        """
        cells_name = cells if isinstance(cells, str) else 'custom'
        cell_ids = self.select_cells(cells)  # grab cells to use
        raster_use = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)[cell_ids]
        baseline_rates = self.event_rates[cell_ids]

        tuning_curves = gen_motion_tuning_curve(raster_use)

        # hopefully future proof for rasters as either a list (as developed)
        ncells = len(raster_use) if type(raster_use) == list else raster_use.shape[0]
        # nevents, nframes = raster_use[0].shape

        nplots = np.ceil(ncells/25).astype('int')
        fig_array = []
        for plot in range(nplots):
            fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)
            fig.set_size_inches([12, 6.9])
            fig.suptitle(self.session['mouse'] + ' ' + self.session['arena'] + ' day ' +
                         str(self.session['day']) + ': ' + cells_name + ' cells: plot ' + str(plot))

            range_use = slice(25*plot, np.min((25*(plot + 1), ncells)))

            for ida, (raster, curve, cell_id, bs_rate, a) in enumerate(zip(raster_use[range_use], tuning_curves[range_use],
                                                                       cell_ids[range_use], baseline_rates[range_use],
                                                                       ax.reshape(-1))):

                # Figure out whether or not to label things - only get edges to keep things clear
                labelx = True if ida >= 20 else False  # Label bottom row
                labely = True if ida in [0, 5, 10, 15, 20] else False  # Label left side
                labely2 = True if ida in [4, 9, 14, 19, 24] else False  # Label right side

                plot_raster(raster, cell_id=cell_id, sig_bins=None, bs_rate=bs_rate, y2scale=0.2, events=events,
                            labelx=labelx, labely=labely, labely2=labely2, sr_image=self.sr_image, ax=a)

            fig_array.append(fig)

        return np.asarray(fig_array)

    def select_events(self, events):
        """Quickly get the appropriate cells and event times to use"""

        # Get appropriate events
        if events == 'freeze_onset':
            event_starts = self.freezing_times[:, 0]
        elif events == 'move_onset':
            event_starts = self.freezing_times[:, 1]

        return event_starts


class MotionTuningMultiDay:
    def __init__(self, mouse: str, arena: str or list, days: list = [-1, 4, 1, 2], events: str = 'freeze_onset',
                 buffer_sec: tuple = (2, 2), **kwargs):
        """
        Create class for tracking motion or freezing tuning of cells across days.
        :param mouse: str of form 'Marble##'
        :param arena: str in ['Open', 'Shock'] or list matching len(days)
        :param days: int in [-2, -1, 0, 4, 1, 2, 7] though day 0 generally not analyzed in 'Shock' due to short
        (60 sec) recording time
        :param buffer_sec: tuple with seconds before/after freeze start to calculated peri-event rasters
        :param **kwargs: see MotionTuning.gen_pe_rasters - inputs for calculating freezing.
        """
        self.mouse = mouse
        self.days = days
        self.events = events

        # Make arena into list below if only one is specified
        if isinstance(arena, str):
            self.arenas = [arena for _ in days]
        else:
            assert len(arena) == len(days), 'Length of arena and days inputs must match'
            self.arenas = arena

        # Dump all days into a dictionary
        self.motion_tuning = {'Open': dict.fromkeys(days), 'Shock': dict.fromkeys(days)}
        self.rois = {'Open': dict.fromkeys(days), 'Shock': dict.fromkeys(days)}
        for arena, day in zip(self.arenas, days):
            self.motion_tuning[arena][day] = MotionTuning(mouse, arena, day, buffer_sec=buffer_sec, **kwargs)  # Get motion tuning for each day above.
            self.motion_tuning[arena][day].gen_pe_rasters(events=events, buffer_sec=buffer_sec)  # generate freeze_onset rasters by default
            self.motion_tuning[arena][day].get_prop_tuned(buffer_sec=buffer_sec, **kwargs)
            self.rois[arena][day] = helpers.get_ROIs(mouse, arena, day)

        # Initialize map between sessions
        self.map = {'map': None, 'base_day': None, 'base_arena': None}

    def get_prop_tuned(self, **kwargs):
        """
        Gets proportion of cells tuned to freezing or movement onset on each day
        :param kwargs: parameters/inputs to MotionTuning.get_sig_neurons() to determine freeze and motion related tuning.
        :return:
        """
        prop_tuned = []
        for arena, day in zip(self.arenas, self.days):
            prop_tuned.append(self.motion_tuning[arena][day].get_prop_tuned(events=self.events, **kwargs))

        return np.asarray(prop_tuned)

    def plot_raster_across_days(self, cell_id, base_arena='Shock', base_day=1, alpha=0.01, label_all=True,
                                labelx=True, ax=None, batch_map=True, plot_ROI=True, label_fig=True,
                                smooth_sec=False, **kwargs):
        """Plots a cell's peri-event raster on base_day and tracks backward/forward to all other days in object.
        e.g. if you have a freezing cell emerge on day1 and want to see what it looked like right before/after,
        use base_day=1

        :param cell_id: int, cell to plot on base_day and track forward/backward to other days_plot
        :param base_arena: str in ['Open', 'Shock']
        :param base_day: int
        :param alpha: value to use for calculating and plotting significant bins
        :param labelx: bool
        :param ax: axes to plot into, default = create new figure with 1 x len(days_plot) subplots
        :param batch_map: use batch map for registering neurons across days as opposed to direct session-to-session reg.
        :param plot_ROI: bool, True (default) = plot ROI shape at bottom row.
        :param **kwargs: to seaborn.heatmap (most likely need to set rasterized=True)
        :return:
        """

        days = self.days
        arenas = self.arenas
        events = self.events

        # Set up figure if not specified
        if ax is None:
            if not plot_ROI:  # one row only if no ROI plotting
                fig, ax = plt.subplots(1, len(days))
                fig.set_size_inches([2.25 * len(days), 2.75])
            else:  # set up rows for plotting rois below - this format keeps roi plots nice and square(ish)
                fig = plt.figure(figsize=[2.25 * len(days), 6])
                gs = gridspec.GridSpec(4, len(days))
                ax, axroi = [], []
                for idd, _ in enumerate(days):
                    ax.append(fig.add_subplot(gs[0:2, idd]))
                    axroi.append(fig.add_subplot(gs[3, idd]))
        else:
            fig = ax.reshape(-1)[0].figure

        # First get map between days
        # reg_id = []  # initialize ids of registered cells
        # for idd, (arena, day) in enumerate(zip(arenas, days)):
        #     neuron_map = pfs.get_neuronmap(self.mouse, base_arena, base_day, arena, day,
        #                                    batch_map_use=batch_map)
        #     reg_id.append(neuron_map[cell_id])  # Get neuron to plot
        self.assemble_map(base_day=base_day, base_arena=base_arena, batch_map=batch_map)  # Create/get neuron map
        # Get column to use for base session
        base_id = np.where([base_day == day and base_arena == arena for arena, day in
                            zip(self.arenas, self.days)])[0][0]
        reg_id = self.map['map'][cell_id]

        # Identify last good neuron for plotting purposes
        last_good_neuron = np.where(np.asarray(reg_id) > 0)[0].max()

        # Now loop through and plot everything!
        ylabel_added = False
        for idd, (arena, day, id_plot) in enumerate(zip(arenas, days, reg_id)):

            if id_plot >= 0:  # only plot if valid mapping between neurons
                raster_plot = self.motion_tuning[arena][day].pe_rasters[events][id_plot]  # get raster
                bs_rate = self.motion_tuning[arena][day].event_rates[id_plot]  # get baseline rate
                sig_bins = np.where(self.motion_tuning[arena][day].sig[events]['pval'][id_plot] < alpha)[0]
                sr_image = self.motion_tuning[arena][day].sr_image

                if label_all:
                    labelx, labely, labely2 = True, True, True
                else:
                    labely = True if idd == 0 or not ylabel_added else False  # label y if on left side
                    labely2 = True if idd == last_good_neuron else False  # label y2 if on right side

                # plot rasters
                _, secax = plot_raster(raster_plot, cell_id=id_plot, bs_rate=bs_rate, events=events,
                                       labelx=labelx, labely=labely, labely2=labely2, ax=ax[idd],
                                       sig_bins=sig_bins, sr_image=sr_image, smooth_sec=smooth_sec, **kwargs)
                ax[idd].set_title(arena + ' Day ' + str(day) + '\n Cell ' + str(id_plot))
                if idd == base_id:  # Make title bold if base day
                    ax[idd].set_title(ax[idd].get_title(), fontweight='bold')

                # Clean it up
                helpers.set_ticks_to_lim(ax[idd])
                ylabel_added = True  # don't label any more y axes...

                if plot_ROI:
                    pfs.plot_ROI_centered(self.rois[arena][day][id_plot], ax=axroi[idd])

            else:  # label things if there is no neuron detected on that day
                ax[idd].text(0.1, 0.5, 'Not detected')
                ax[idd].set_xticks([])
                ax[idd].set_yticks([])
                sns.despine(ax=ax[idd], left=True, bottom=True)

                if plot_ROI:  # Clean up bottom plots if no neuron
                    axroi[idd].set_xticks([])
                    axroi[idd].set_yticks([])
                    sns.despine(ax=axroi[idd], left=True, bottom=True)

        if label_fig:
            fig.suptitle(self.mouse + ': Across Days')

        return ax

    def assemble_map(self, base_day: int in [-2, -1, 4, 1, 2, 7], base_arena: str in ['Open', 'Shock'],
                     batch_map: bool = False):
        """
        Assembles all neuron mappings from base day to other days in self.days
        :param base_day:
        :param base_arena:
        :param batch_map: bool, how to register neurons across day: via batch map (False)
        or direct mapping (True, default).
        :return:
        """

        if self.map['map'] is not None and self.map['base_day'] == base_day and \
                self.map['base_arena'] == base_arena:
            # use previously calculated map.
            map = self.map['map']
        else:
            print('Assembling neuron map for base_day=' + str(base_day) + ' and base_arena=' + base_arena)
            # Loop through each session and assemble map
            map = []
            for id, (arena, day) in enumerate(zip(self.arenas, self.days)):
                map.append(pfs.get_neuronmap(self.mouse, base_arena, base_day, arena, day,
                                               batch_map_use=batch_map))

            self.map['map'] = np.asarray(map).swapaxes(0, 1)
            self.map['base_arena'] = base_arena
            self.map['base_day'] = base_day

        return map

    def get_pval_across_days(self, base_day: int in [-2, -1, 0, 4, 1, 2, 7], adj_bins: int = 1):
        """
        Grab mean pval at peak of tuning curve and +/- adj_bins next to the peak.  Evaluates
        the extent to which a cell maintains its freeze or motion related tuning across days.
        Or should this just evaluate if the cell has significant tuning or not the next day?
        :return:
        """
        pass

    def get_tuning_loc_diff(self, cell_id: int, base_day: int in [-2, -1, 0, 4, 1, 2, 7] = 1,
                            base_arena: str in ['Shock', 'Open'] = 'Shock', smooth_window: int = 4):
        """
        Grab location of peak tuning curve and track across days! Finer grained
        look at how well a cell maintains its freeze or motion related tuning.
        :return: locs: location of peak tuning after smoothing
        :return: event_rates: event rate at peak tuning location (mean of all bins within smoothing window)
        :return: pvals: pval at peak tuning location (mean of all bins within smoothing window)
        :return: corr: spearman corr of tuning curve in dict with key 'corrs' and 'pvals'
        """

        # Get map between neurons
        self.assemble_map(base_day=base_day, base_arena=base_arena)  # Create/get neuron map
        reg_id = self.map['map'][cell_id]

        # Get column to use for base session
        base_id = np.where([base_day == day and base_arena == arena for arena, day in
                            zip(self.arenas, self.days)])[0][0]

        # pre-allocate arrays
        locs, event_rates = np.ones(len(self.days))*np.nan, np.ones(len(self.days))*np.nan
        pvals, is_tuned = np.ones(len(self.days))*np.nan, np.ones(len(self.days))*np.nan

        window_half = smooth_window/2  # Get half of smoothing window

        # Loop through each day, compare to base_day, and get peak tuning location and pval
        tuning_curve_all = []
        for idd, (arena, day, id) in enumerate(zip(self.arenas, self.days, reg_id)):

            if id >= 0:  # only calculate if valid mapping between neurons
                tuning_curves = gen_motion_tuning_curve(self.motion_tuning[arena][day].pe_rasters[self.events])
                tuning_curve_all.append(tuning_curves[id])

                # Get max event rate and its location (time)
                loc_bins, event_rates[idd] = get_tuning_max(tuning_curves[id], window=smooth_window)
                locs[idd] = loc_bins/self.motion_tuning[arena][day].sr_image  # convert to image

                # Grab pvalues for tuning curve
                p_use = self.motion_tuning[base_arena][day].sig[self.events]['pval'][id]

                # Calculate mean p-value along the curve at the location of the peak (within the smoothing window)
                pvals[idd] = np.mean(p_use[int(locs[idd]-window_half):int(locs[idd]+window_half)])

                # Figure out if it is a freeze cell on that day
                is_tuned[idd] = id in self.motion_tuning[base_arena][day].get_sig_neurons()
            else:
                tuning_curve_all.append([])

        # Now run correlation between all days
        corrs, pcorrs = np.ones(len(self.days))*np.nan, np.ones(len(self.days))*np.nan
        for idd, (curve, id) in enumerate(zip(tuning_curve_all, reg_id)):

            if id >= 0:  # only get legit mapping values
                corrs[idd], pcorrs[idd], _ = pfs.spearmanr_nan(curve, tuning_curve_all[base_id])

        corr = {'corrs': corrs, 'pvals': pcorrs}

        return locs, event_rates, pvals, corr, is_tuned


def snake_plot(rasters, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    peak_id = np.array([helpers.allmax(rast) for rast in rasters])
    sort_ids = np.argsort(peak_id)
    sns.heatmap(rasters[sort_ids], ax=ax)

    return ax, peak_id


def freeze_group_snake_plot(group, arena, day, buffer_sec=(2, 2), sr_match=20, ax=None, plot=True):
    """Plot freeze-tuned cells for each group/day"""

    # Set up times
    times = np.arange(-buffer_sec[0], buffer_sec[1], 1 / sr_match)

    freeze_rasts_mean = []
    for mouse in group:
        MD1 = MotionTuning(mouse, arena, day, buffer_sec=buffer_sec)
        MD1.gen_pe_rasters(buffer_sec=buffer_sec)
        freeze_cells = MD1.select_cells('freeze_fine', buffer_sec=buffer_sec)
        rast_use = MD1.pe_rasters['freeze_onset'][freeze_cells]

        if MD1.sr_image != sr_match:  # interpolate values if sample rate doesn't match to make data compatible
            rast_mean_interp = []
            times_sr = np.arange(-buffer_sec[0], buffer_sec[1], 1 / MD1.sr_image)
            for act in rast_use.mean(axis=1):
                rast_mean_interp.append(np.interp(times, times_sr, act))
            freeze_rasts_mean.append(rast_mean_interp)
        else:
            freeze_rasts_mean.append(rast_use.mean(axis=1))

    freeze_rasts_mean_comb = np.concatenate(freeze_rasts_mean, axis=0)

    if plot:
        ax, peak_id = snake_plot(freeze_rasts_mean_comb, ax=ax)
        ax.set_xticks([0, int(len(times) / 2), len(times)])
        ax.set_xticklabels([-buffer_sec[0], 0, buffer_sec[1]])
    else:
        ax = None
        peak_id = np.array([helpers.allmax(rast) for rast in freeze_rasts_mean_comb])

    return ax, peak_id, times


class TuningStability:
    """Class to examine within arena stability of tuning curves across days"""
    def __init__(self, arena, events, alpha):
        self.arena = arena
        self.events = events
        self.alpha = alpha
        self.days = [-1, 4, 1, 2]

        # First, try loading in previously saved class
        file_use = path.join(err.working_dir, events + '_' + arena + '_tuning_across_days_alpha' +
                             str(alpha).replace('.', '_') + '.pkl')
        if path.exists(file_use):
            with open(file_use, 'rb') as f:
                self.tuning_stability = load(f)
        else:  # calculate everything if not already saved
            self.tuning_stability = assemble_tuning_stability(arena=arena, events=events, alpha=alpha)
            with open(file_use, 'wb') as f:  # save it!
                dump(self.tuning_stability, f)

        # Double check tuning_stability fields are compatible with save name (backwards compatibility)
        assert events == self.tuning_stability['events'], '"events" field incompatible with saved value in ' + file_use
        assert arena == self.tuning_stability['base_arena'], \
            '"arena" field incompatible with saved value in ' + file_use
        assert self.days == self.tuning_stability['days'], '"days" field incompatible with saved value in ' + file_use

    def plot_prop_tuned(self, group='Learners', plot_by='mouse', ax=None):
        """This will plot the proportion of total cells that are tuned to freeze/motion onset events"""
        pass

    def get_off_ratio(self, group, base_day):
        """Determine how many freeze or motion related cells are turning off from base day to other days"""
        locs_ = []
        for locs in self.tuning_stability[group][base_day]['locs']:
            locs_.append(np.isnan(locs).sum(axis=0) / locs.shape[0])
        off_ratio = np.asarray(locs_)

        return off_ratio

    def get_overlap_ratio(self, group, base_day):
        """Determines the probability a motion-tuned cell on base day retains that tuning on a different day """
        tuned = []
        for is_tuned in self.tuning_stability[group][base_day]['is_tuned']:
            tuned.append(np.nansum(is_tuned, axis=0) / is_tuned.shape[0])

        return np.asarray(tuned)

    def off_ratio_to_df(self, base_day):
        """Send all off ratio data to a nicely organized dataframe. Also sends overlap ratio"""
        # First loop through and get all off data
        df_list = []
        for exp_group, group in zip(['Control', 'Control', 'ANI'], ['Learners', 'Nonlearners', 'ANI']):
            off_ratio = self.get_off_ratio(group, base_day)  # NRK todo: this should move into loop below!!! Leave for now.
            overlap_ratio = self.get_overlap_ratio(group, base_day)
            # Now assign appropriate day and mouse and group to each data point
            mouse, group_names, day, base, exp_group_names = [], [], [], [], []
            for idr, (ratio, overlap) in enumerate(zip(off_ratio, overlap_ratio)):
                day.extend(self.days)
                mouse.extend(np.ones_like(ratio, dtype=int)*idr)
                base.extend(np.ones_like(ratio)*base_day)
                group_names.extend([group for _ in ratio])
                exp_group_names.extend([exp_group for _ in ratio])

            df_temp = pd.DataFrame({'Exp Group': exp_group_names, 'Group': group_names, 'Mouse': mouse,
                                    'Base Day': base, 'Day': day, 'Off Ratio': off_ratio.reshape(-1),
                                    'Overlap Ratio': overlap_ratio.reshape(-1)})
            df_list.append(df_temp)

        df_all = pd.concat(df_list)

        return df_all

    def metric_to_df(self, base_day, metric, delta=False):
        """Send stability metric to a nicely organized dataframe.
        if delta=True it will subtract everything from the base day"""
        # First loop through and get all off data
        df_list = []
        day_bool = np.asarray([d == base_day for d in self.days])
        for exp_group, group in zip(['Control', 'Control', 'ANI'], ['Learners', 'Nonlearners', 'ANI']):
            metric_use = self.tuning_stability[group][base_day][metric]
            # Now assign appropriate day and mouse and group to each data point
            mouse, group_names, day, base, exp_group_names = [], [], [], [], []
            for idr, met in enumerate(metric_use):
                ncells = met.shape[0]
                day.extend(np.matlib.repmat(self.days, ncells, 1).reshape(-1))
                mouse.extend(np.ones_like(met, dtype=int).reshape(-1)*idr)
                base.extend(np.ones_like(met, dtype=int).reshape(-1)*base_day)
                group_names.extend([group] * (ncells*len(self.days)))
                exp_group_names.extend([exp_group] * (ncells*len(self.days)))

            if not delta:
                metric_final, met_name = np.vstack(metric_use), metric
            elif delta:  # subtract out base day values to get delta
                metric_final = np.vstack(metric_use) - np.vstack(metric_use)[:, day_bool]
                met_name = 'Delta' + metric
            df_temp = pd.DataFrame({'Exp Group': exp_group_names, 'Group': group_names, 'Mouse': mouse,
                                    'Base Day': base, 'Day': day, met_name: np.vstack(metric_final).reshape(-1)})
            df_list.append(df_temp)

        df_all = pd.concat(df_list)

        return df_all

    def plot_off_ratio(self, base_day=4, group='Learners', plot_by='mouse', ax=None):
        """Plots the probability a event-tuned cell turns off from one session to the next"""

        # Set up axes to plot into
        if ax is None:
            fig, ax = plt.subplots()

        # Get probability a cells turns off for each mouse - should be a nan if it does.
        assert plot_by in ('mouse', 'group'), '"plot_by" must be either "mouse" or "group"'
        locs_ = []
        if plot_by == 'mouse':
            # Get proportion of cells turning off for each mouse
            for locs in self.tuning_stability[group][base_day]['locs']:
                locs_.append(np.isnan(locs).sum(axis=0)/locs.shape[0])
            off_ratio = np.asarray(locs_)
        elif plot_by == 'group':
            for locs in self.tuning_stability[group][base_day]['locs']:
                locs_.append(locs)
            off_ratio = np.isnan(np.asarray(locs_)).sum(axis=0)/np.asarray(locs_).shape[0]
            off_ratio = off_ratio.reshape(1, -1)  # Make this a 1 x ndays array

        for ratio in off_ratio:
            ax.plot(list(range(len(self.days))), ratio)

        ax.set_xticks(list(range(len(self.days))))
        ax.set_xticklabels([str(day) for day in self.days])
        ax.set_xlabel('Session')
        ax.set_ylabel('Off proportion')
        ax.set_title(group)
        sns.despine(ax=ax)

        return ax

    def plot_off_ratio_by_group(self, base_day, group='Exp Group'):
        """
        Plot off ratio in bar/scatter format between groups
        :param base_day:
        :param group:
        :return:
        """
        # Set up color palette
        assert group in ['Group', 'Exp Group'], 'group must be "Exp Group" or "Group"'
        if group == 'Exp Group':
            pal_use = [(0, 0, 0), (0, 1, 0)]
            pal_use2 = [(0.2, 0.2, 0.2, 0.1), (0, 1, 0, 0.1)]  # Necessary to make sure scatterplot visible over bar
        elif group == 'Group':
            pal_use, pal_use2 = 'Set2', 'Set2'

        df = self.off_ratio_to_df(base_day)  # Make data into a dataframe
        fig, ax = plt.subplots(1, 2)  # set up figures
        fig.set_size_inches((12.9, 4.75))
        # Plot scatterplot
        sns.stripplot(x='Day', y='Off Ratio', data=df, hue=group, dodge=True, ax=ax[0], palette=pal_use,
                      order=[-1, 4, 1, 2])

        # This is necessary to prevent duplicated in legend
        group_rows = df.loc[:, group].copy()
        group_rows_ = ["_" + row for row in df[group]]
        df.loc[:, group] = group_rows_

        # Now plot bars overlaying
        sns.barplot(x='Day', y='Off Ratio', data=df, hue=group, dodge=True, ax=ax[0], palette=pal_use2,
                    order=[-1, 4, 1, 2])

        # Clean up and label
        sns.despine(ax=ax[0])
        ax[0].set_title('Freeze cells')
        ax[0].set_xlabel('Session Before/After')

        df.loc[:, group] = group_rows  # set labels back to normal

        # Now do stats!
        days_test = [-1, 4, 2] if base_day == 1 else [-1, 1, 2]
        for ycoord, day in zip([0.3, 0.5, 0.7], days_test):
            ctrl = df[np.bitwise_and(df['Exp Group'] == 'Control', df['Day'] == day)]
            ani = df[np.bitwise_and(df['Exp Group'] == 'ANI', df['Day'] == day)]
            stat, pval = stats.ttest_ind(ctrl['Off Ratio'], ani['Off Ratio'])
            print(f'2sided t-test Day {day} : pval={pval:.3g} tstat={stat:.3g}')
            ax[1].text(0.1, ycoord, f'2sided t-test Day {day} : pval={pval:.3g} tstat={stat:.3g}')
        sns.despine(ax=ax[1], left=True, bottom=True)
        ax[1].axis(False)

        return fig, ax

    def plot_metric_stability(self, base_day=4, group='Learners', metric_plot='event_rates', ax=None):
        """Plots a metric across days for a single group"""
        sr_image = 20  # Make this an input or come from that animal's class since one animal has sr=10 Hz

        # Figure out ylabel
        metrics = ["locs", "event_rates", "pvals", "corr_corrs", "corr_pvals"]
        metric_labels = [r'$\Delta_t$', r'$\Delta{p}_{event}$', 'p at peak', r'$\rho$', r'$p_{\rho}$']
        met_ind = np.where([metric_plot == met for met in metrics])[0][0]
        met_label = metric_labels[met_ind]

        # Assemble cross-day stability metrics from list into pandas array
        a = pd.concat([pd.DataFrame.from_dict(_) for _ in self.tuning_stability[group][base_day][metric_plot]])
        # Rename rows to match each day.
        b = a.rename({key: value for key, value in enumerate(self.days)}, axis=1).copy()

        # Set up axes to plot into
        if ax is None:
            fig, ax = plt.subplots()

        # Subtract base day value if looking at tuning curve location or event rate across days
        if metric_plot in ['locs', 'event_rates']:
            data_use = b.subtract(b[base_day], axis='rows')
        else:
            data_use = b

        sns.stripplot(data=data_use, ax=ax)
        ax.plot(ax.get_xticks(), data_use.mean())  # plot mean
        # if metric_plot == 'locs':  # Adjust y-axis for time
        #     ax.set_yticklabels([str(ytick / sr_image) for ytick in ax.get_yticks()])
        ax.set_ylabel(met_label)  # Label metric appropriately
        ax.set_xlabel('Session')
        ax.set_title(group + ': Base_day = ' + str(base_day))

        sns.despine(ax=ax)

    def plot_metric_stability_by_group(self, base_day: int, metric_plot: str, delta: bool = True,
                                       days_plot: list or int or None = None,
                                       group_by: str in ['Group', 'Exp Group'] = 'Group', ax=None, **kwargs):


        # plotting info
        metrics = ['locs', 'event_rates', 'pvals', 'corr_corrs', 'corr_pvals']
        metric_labels_delta = [r'$\Delta_t$', r'$\Delta{ER}_{peak}$ (1/s)', 'p at peak', r'$\rho$', r'$p_{\rho}$']
        metric_labels = [r'$t (s)$', r'$ER_{peak} (1/s)$', 'p at peak', r'$\rho$', r'$p_{\rho}$']
        pal_use, pal_use_bar = get_palettes(group_by)  # Get colors to plot into

        # First, get delta in event rates across days as a dataframe for day1 cells
        if days_plot is None:  # Automatically pick days before/after
            days = [-2, -1, 4, 1, 2, 7]
            base_ind = np.where([base_day == d for d in days])[0][0]
            days_plot = days[slice(base_ind-1, base_ind+2, 2)]
        elif days_plot is int:
            days_plot = [days_plot]

        # Now send tuning data to dataframe and pick out only days of interest
        df_full = self.metric_to_df(base_day, metric_plot, delta=delta)
        df = df_full[[d in days_plot for d in df_full['Day']]]  # Keep only days indicated in days_plot

        # set up figure
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches((12.9, 4.75))
        else:
            fig = ax[0].figure
        met_name = metric_plot if not delta else 'Delta' + metric_plot

        # Plot scatter
        sns.stripplot(x='Day', y=met_name, data=df.reset_index(), hue=group_by, dodge=True, order=days_plot,
                      palette=pal_use, ax=ax[0], **kwargs)

        # This is the only easy way I could figure out to NOT duplicate labels in the legend
        group_rows = df.loc[:, group_by].copy()  # This generates warnings about chained indexing for some reason
        group_rows_ = ["_" + row for row in df[group_by]]
        df.loc[:, group_by] = group_rows_

        # Plot overlaying bar graph
        sns.barplot(x='Day', y=met_name, data=df.reset_index(), hue=group_by, dodge=True, order=days_plot,
                    palette=pal_use_bar, facecolor=(1, 1, 1, 0), edgecolor=(1, 1, 1, 0), ax=ax[0])

        # Cleanup
        ax[0].legend(loc='upper right')
        df.loc[:, group_by] = group_rows
        sns.despine(ax=ax[0])

        # Label y-axis nicely
        met_ind = np.where([met == metric_plot for met in metrics])[0][
            0]  # find out index for which metric you are plotting
        met_label = metric_labels_delta[met_ind] if delta else metric_labels[met_ind]
        ax[0].set_ylabel(met_label)
        ax[0].set_xlabel('Session Before/After')

        # Label title
        ax[0].set_title('Freeze cells on Day ' + str(base_day))

        # Now get stats and print out
        ycoord = 0.9
        fontdict = {'size': 3}
        if group_by == 'Group':
            for day in days_plot:
                df_day = df[df['Day'] == day]
                astats = pg.anova(data=df_day, dv='Deltaevent_rates', between='Group')
                posthoc_stats = pg.pairwise_tukey(data=df_day, dv='Deltaevent_rates', between='Group')
                ax[1].text(0.1, ycoord + 0.2, f'Day {day} anova and post-hoc tukey test stats', fontdict=fontdict)
                ax[1].text(0.1, ycoord + 0.1,
                           f"n = {[np.sum(~np.isnan(df_day[df_day['Group'] == gname]['Deltaevent_rates'])) for gname in df_day['Group'].unique()]} for {df_day['Group'].unique()}",
                           fontdict=fontdict)
                ax[1].text(0.1, ycoord, str(astats), fontdict=fontdict)
                ax[1].text(0.1, ycoord - 0.5, str(posthoc_stats), fontdict=fontdict)
                ycoord -= 0.8

        else:
            print('Stats not yet enabled for "Exp Group" plotting')
            ax[1].text(0.1, 0.5, 'Stats not yet enabled for "Exp Group" plotting')

        ax[1].axis(False)

        return fig, ax


class TuningGeneralization:
    """Class to examine tuning curve generalization between arenas"""
    def __init__(self, events, alpha):
        pass


def assemble_tuning_stability(arena='Shock', events='freeze_onset', alpha=0.01):
    """Get freeze cell stability tuning across days - probably should live in TuningStability class"""
    mice_groups = [err.learners, err.nonlearners, err.ani_mice_good]
    group_names = ['Learners', 'Nonlearners', 'ANI']
    base_arena = arena
    days = [-1, 4, 1, 2]  # Start with post-learning cells + one pre-learning day
    base_days = [4, 1]

    # Pre-allocate dictionary for keeping track of everything
    tuning_stability = dict.fromkeys(group_names)
    for group in group_names:
        tuning_stability[group] = dict.fromkeys(base_days)
        for base_day in base_days:
            tuning_stability[group][base_day] = {'locs': [], 'event_rates': [], 'pvals': [], 'corr_corrs': [],
                                                 'corr_pvals': [], 'is_tuned': []}

    # Now run everything
    for gname, group in zip(group_names, mice_groups):
        for mouse in group:
            print('Running Mouse ' + mouse)
            if mouse == 'Marble14':
                print('Running Marble14. MAKE SURE TO ADJUST all CODE FOR 10Hz FRAME RATE!!!')
            mmd = MotionTuningMultiDay(mouse, 'Shock', days=days, events=events)
            for base_day in [4, 1]:

                # Get sig neurons here!
                sig_neurons = mmd.motion_tuning[base_arena][base_day].get_sig_neurons(alpha=alpha)

                # Start up list for each mouse/day pair
                locs_all, event_rates_all, pvals_all, corr_corrs_all, corr_pvals_all = [], [], [], [], []
                is_tuned_all = []
                for sig_neuron in sig_neurons:
                    # Track each neuron across days!
                    locs, event_rates, pvals, corr, is_tuned = mmd.get_tuning_loc_diff(sig_neuron, base_day=base_day,
                                                                             base_arena=base_arena)
                    locs_all.append(locs)
                    event_rates_all.append(event_rates)
                    pvals_all.append(pvals)
                    corr_corrs_all.append(corr['corrs'])
                    corr_pvals_all.append(corr['pvals'])
                    is_tuned_all.append(is_tuned)

                # Now add this into dictionary
                for var, name in zip([locs_all, event_rates_all, pvals_all, corr_corrs_all, corr_pvals_all, is_tuned_all],
                                     ["locs", "event_rates", "pvals", "corr_corrs", "corr_pvals", "is_tuned"]):
                    tuning_stability[gname][base_day][name].append(np.asarray(var))

    tuning_stability['days'] = days
    tuning_stability['events'] = events
    tuning_stability['base_arena'] = 'Shock'

    return tuning_stability


def get_freezing_times(mouse, arena, day, zero_start=True, **kwargs):
    """Identify chunks of frames and timestamps during which the mouse was freezing in behavioral movie!

    :param mouse: str
    :param arena: 'Open' or 'Shock'
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param zero_start: boolean, True (default) sets first behavioral time point to 0. Use if your imaging and behavioral
    data are already aligned!
    :param kwargs: Freezing parameters to use. See er_plot_functions.detect_freezing()
    :return: freezing_epochs: list of start and end indices of each freezing epoch in behavioral video
             freezing_times: list of start and end times of each freezing epoch
    """
    dir_use = erp.get_dir(mouse, arena, day)

    # Get freezing times
    freezing, velocity, video_t = erp.detect_freezing(str(dir_use), arena=arena, return_time=True, **kwargs)
    assert video_t.shape[0] == (velocity.shape[0] + 1), 'Mismatch between time and velocity arrays'
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # Set first tracking time to 0 if specified
    if zero_start:
        video_t = video_t - video_t[0]

    # convert freezing indices to timestamps
    freezing_indices = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_indices]

    return np.asarray(freezing_indices), np.asarray(freezing_times)


def align_freezing_to_PSA(PSAbool, sr_image, freezing, video_t, PSAaligned=True):
    """
    Align freezing times to neural data.
    :param PSAbool: nneurons x nframes_imaging boolean ndarray of putative spiking activity
    :param sr_image: frames/sec (int)
    :param freezing: output of er_plot_functions.detect_freezing() function.
    :param video_t: video frame timestamps, same shape as `freezing`
    :param PSAaligned: bool, True (default) indicates PSAbool data is aligned to behavioral data and have the same
    start time.
    :return: freeze_bool: boolean ndarray of shape (nframes_imaging,) indicating frames where animals was freezing.
    """

    # First get imaging parameters and freezing in behavioral video timestamps
    nneurons, nframes = PSAbool.shape
    freezing_epochs = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_epochs]

    # Set up boolean to match neural data shape
    freeze_bool = np.zeros(nframes, dtype='bool')
    PSAtime = np.arange(0, nframes)/sr_image
    if PSAaligned:  # Make PSA start at behavioral video start time if data is already aligned
        PSAtime = PSAtime + video_t[0]

    # Interpolate freezing times in video time to imaging time
    for freeze_time in freezing_times:
        try:
            freeze_bool[np.bitwise_and(PSAtime >= freeze_time[0], PSAtime < freeze_time[1])] = True
        except IndexError:
            print('debugging')

    return freeze_bool


def freeze_event_rate(PSAbool, freeze_bool):
    """
    Calculate event rate during freezing times only.
    :param PSAbool: nneurons x nframes_imaging boolean ndarray of putative spiking activity
    :param freeze_bool: boolean ndarray of shape (nframes_imaging,) indicating frames where animals was freezing.
    Get from function `align_freezing_to_PSA`.
    :return: event_rate_freezing: ndarray of each neuron's event rate during freezing epochs.
    """
    event_rate_freezing = PSAbool[:, freeze_bool].sum(axis=1) / freeze_bool.sum()

    return event_rate_freezing


def move_event_rate(PSAbool, freeze_bool):
    """
        Calculate event rate during motinon (non-freezing) times only.
        :param PSAbool: nneurons x nframes_imaging boolean ndarray of putative spiking activity
        :param freeze_bool: boolean ndarray of shape (nframes_imaging,) indicating frames where animals was freezing.
        Get from function `align_freezing_to_PSA`.
        :return: event_rate_moving: ndarray of each neuron's event rate during freezing epochs.
        """
    event_rate_moving = PSAbool[:, np.bitwise_not(freeze_bool)].sum(axis=1) / np.bitwise_not(freeze_bool).sum()

    return event_rate_moving


def get_freeze_bool(mouse, arena, day, **kwargs):
    """ Gets boolean of freezing times aligned to neural data!"""
    # First get directory and neural data
    dir_use = erp.get_dir(mouse, arena, day)
    PF = pf.load_pf(mouse, arena, day)

    # Now get behavioral timestamps and freezing times
    freezing, velocity, video_t = erp.detect_freezing(str(dir_use), arena=arena, return_time=True, **kwargs)
    assert video_t.shape[0] == (velocity.shape[0] + 1), 'Mismatch between time and velocity arrays'
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # Now align freezing to neural data!
    freeze_bool = align_freezing_to_PSA(PF.PSAbool_align, PF.sr_image, freezing, video_t, PSAaligned=True)

    return PF.PSAbool_align, freeze_bool


def get_freeze_ratio(mouse, arena, day, **kwargs):
    """Gets freeze ratio. **kwargs see get_freeze_bool and er_plot_functions.detect_freezing"""
    _, freeze_bool = get_freeze_bool(mouse, arena, day, **kwargs)

    return freeze_bool.sum() / len(freeze_bool)


def motion_modulation_index(mouse, arena, day, **kwargs):
    """ Calculate motion modulation index (MMI): difference/sum of event rates during motion and freezing (1 = only
    active during motion, -1 = only active during freezing)

    :param mouse: str
    :param arena: str ('Shock', 'Open')
    :param day: int (-2, -1, 0, 4, 1, 2, 7)
    :param kwargs: see er_plot_functions.detect_freezing() for relevant arguments, most notably `velocity_threshold` and
    `min_freeze_duration`
    :return:
    """
    # First get directory and neural data
    dir_use = erp.get_dir(mouse, arena, day)
    PF = pf.load_pf(mouse, arena, day)

    # Now get behavioral timestamps and freezing times
    video_t = erp.get_timestamps(str(dir_use))
    freezing, velocity = erp.detect_freezing(str(dir_use), arena=arena, **kwargs)
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # Now align freezing to neural data!
    freeze_bool = align_freezing_to_PSA(PF.PSAbool_align, PF.sr_image, freezing, video_t)

    # Get moving and freezing event rates and calculate MMI
    event_rate_moving = move_event_rate(PF.PSAbool_align, freeze_bool)
    event_rate_freezing = freeze_event_rate(PF.PSAbool_align, freeze_bool)
    MMI = (event_rate_moving - event_rate_freezing) / (event_rate_moving + event_rate_freezing)

    return MMI


def calc_sig_modulation(mouse, arena, day, nperms=1000, **kwargs):
    """ Calculates how much each cell is modulated by moving, freezing, and a combination
    (Motion Modulation Index = MMI).  Gives p value based on circularly permuting neural activity.
    Rough - does not consider cells that might predict freezing or motion, only considers whole epoch of motion
    or freezing.
    :param mouse: str
    :param arena: str ('Shock' or 'Open')
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param nperms: 1000 = default
    :param kwargs: args to eraser_plot_functions.detect_freezing for calculating freezing epochs.
    :return:
    """
    PSAbool, freeze_bool = get_freeze_bool(mouse, arena, day, **kwargs)

    # Get moving and freezing event rates and calculate MMI
    event_rate_moving = move_event_rate(PSAbool, freeze_bool)
    event_rate_freezing = freeze_event_rate(PSAbool, freeze_bool)
    MMI = (event_rate_moving - event_rate_freezing) / (event_rate_moving + event_rate_freezing)

    # Now shuffle things up and recalculate everything!
    ER_freeze_shuf, ER_move_shuf, MMI_shuf = [], [], []
    shifts = [np.random.randint(0, PSAbool.shape[1]) for _ in range(nperms)]
    for shift in shifts:
        PSAshuf = np.roll(PSAbool, shift, axis=1)
        ER_move_shuf.append(move_event_rate(PSAshuf, freeze_bool))
        ER_freeze_shuf.append(freeze_event_rate(PSAshuf, freeze_bool))
        MMI_shuf.append((ER_move_shuf[-1] - ER_freeze_shuf[-1]) / (ER_move_shuf[-1] + ER_freeze_shuf[-1]))

    # Make lists into workable arrays
    ER_move_shuf = np.asarray(ER_move_shuf)
    ER_freeze_shuf = np.asarray(ER_freeze_shuf)
    MMI_shuf = np.asarray(MMI_shuf)

    # Now calculate significance here!
    pmove = ((event_rate_moving - ER_move_shuf) < 0).sum(axis=0)/nperms
    pfreeze = ((event_rate_freezing - ER_freeze_shuf) < 0).sum(axis=0)/nperms

    # Note that this is two sided!!! - things with p < 0.05 should be motion modulated,
    # # p > 0.95 should be freeze modulated, need to double check
    pMMI = ((MMI - MMI_shuf) < 0).sum(axis=0)/nperms

    # Dump things into a dictionary for easy access later
    p = {'move': pmove, 'freeze': pfreeze, 'MMI': pMMI}
    ER = {'move': event_rate_moving, 'freeze': event_rate_freezing, 'MMI': MMI}

    return p, ER


def moving_average(arr, n=10):
    """Get a moving average of calcium activity"""
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def bin_array(arr, n=10):
    """Get total counts of calcium activity in n frame bins (10 by default)"""
    return np.add.reduceat(arr, np.arange(0, len(arr), n))


def get_PE_raster(psa, event_starts, buffer_sec=(2, 2), sr_image=20):
    """ Gets peri-event rasters for +/-buffers sec from all event start times in event_starts
    :param psa: activity for one cell at sr_image, frame 0 = time 0 in event_starts
    :param event_starts: list of event start times in seconds.
    :param buffer_sec: float, sec or length 2 array/list of buffer times before/after
    :param sr_image: frame rate for imaging data
    :return:
    """

    if len(buffer_sec) == 1:  # Make into size 2 array if only one int specified
        buffer_sec = [buffer_sec, buffer_sec]

    # Get # frames before/after event to include in raster
    buffer_frames = [int(buffer_sec[0] * sr_image), int(buffer_sec[1] * sr_image)]

    # Exclude any events where the buffer extends beyond the start/end of the neural recording
    first_ok_time = buffer_frames[0]/sr_image
    last_ok_time = (len(psa) - buffer_frames[1])/sr_image
    good_event_bool = np.bitwise_and(np.asarray(event_starts) >= first_ok_time,
                                     np.asarray(event_starts) <= last_ok_time)
    filtered_starts = [start for (start, ok) in zip(event_starts, good_event_bool) if ok]

    raster_list = []
    for start_time in filtered_starts:
        start_id = int(start_time * sr_image)
        raster_list.append(psa[(start_id - buffer_frames[0]):(start_id + buffer_frames[1])])

    # pe_raster = np.asarray(raster_list[1:])
    pe_raster = np.asarray(raster_list)

    return pe_raster


def shuffle_raster(psa, event_starts, buffer_sec=(2, 2), sr_image=20, nperm=1000):
    """Calculates shuffled event rasters by circularly permuting psa.

    :param psa: ndarray of event activity at sr_image
    :param event_starts: list of start times
    :param buffer_sec: before/after times to use to calculate raster, float. default = [2, 2]
    :param sr_image: int, 20 = default
    :param nperm: int, 1000 = default
    :return:
    """

    perms = np.random.permutation(len(psa))[0:(nperm+1)]  # get nperms

    shuffle_raster = []
    for perm in perms:
        psashuf = np.roll(psa, perm)  # permute psa
        shuffle_raster.append(get_PE_raster(psashuf, event_starts, buffer_sec=buffer_sec, sr_image=sr_image))

    return np.asarray(shuffle_raster[1:])


def gen_motion_tuning_curve(pe_rasters):
    """Function to write to generate neural tuning curves at onset or offset of motion.

    :param pe_rasters: 3d ndarray (ncells x nevents x ntimebins)
    :return:
    """
    # Make 3d array if just one raster input
    if len(pe_rasters.shape) == 2:
        pe_rasters = pe_rasters[np.newaxis, :, :]

    tuning_curves = pe_rasters.sum(axis=1)/pe_rasters.shape[1]

    return tuning_curves


def get_tuning_max(tuning_curve: np.ndarray, window: int = 4):
    """
    Finds the bin where the maximum of the tuning curve occurs.
    :param tuning_curve: ndarray, tuning curve for a neuron centered on a freeze or move onset event
    :param window: int, # bins to smooth prior to finding max (default = 5)
    :return:
    """

    curve_smooth = pd.Series(tuning_curve.squeeze()).rolling(window, center=True).mean()

    max_val = curve_smooth.max()

    # If multiple max values take the average of them
    if (npts := np.sum(max_val == curve_smooth)) > 1:
        if npts >= 4:
            print('More than 4 max points found, check and validate code for more points')
        max_loc = np.where(max_val == curve_smooth)[0].mean()
    elif npts == 1:
        print('one point found!')
        max_loc = curve_smooth.argmax()

    return max_loc, max_val


def get_palettes(group: str in ['Group', 'Exp Group']):
    """Returns appropriate color palettes to use for plotting by 'Group' or 'Exp Group' with data points
    and a bar plot"""
    if group == 'Exp Group':
        pal_use = [(0, 0, 0), (0, 1, 0)]
        pal_use_bar = [(0.2, 0.2, 0.2, 0.1), (0, 1, 0, 0.1)]  # Necessary to make sure scatterplot visible over bar
    elif group == 'Group':
        pal_use, pal_use_bar = 'Set2', 'Set2'

    return pal_use, pal_use_bar


def plot_raster(raster, cell_id=None, sig_bins=None, sig_style='r.', bs_rate=None, y2scale=0.25, events='trial',
                labelx=True, labely=True, labely2=True, sr_image=20, ax=None, y2zero=0, cmap='rocket',
                smooth_sec=False, **kwargs):
    #NRK todo: change bs_rate plot to incorporate sample rate. currently too high!!!
    """Plot peri-event raster with tuning curve overlaid.

    :param raster: nevents x nframes array
    :param cell_id: int, cell # to label with
    :param sig_bins: bool, frames with significant tuning to label with *s
    :param sig_style: style to plot significance bins with, default = 'r.'
    :param bs_rate: float, baseline rate outside of events
    :param y2scale: float, 0.2 = default, scale for plotting second y-axis (event rate)
    :param events: str, for x and y labels
    :param labelx: bool
    :param labely: bool
    :param labely2: bool
    :param sr_image: int, default = 20 fps
    :param ax: ax to plot into, default = None -> create new fig
    :param y2zero: location of y2 axis zero point in # trials
    :param **kwargs to heatmap
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([2.5, 3])

    curve = gen_motion_tuning_curve(raster).squeeze()
    if smooth_sec:  # Smooth if specified
        curve = np.convolve(curve, np.ones(int(smooth_sec*sr_image)),
                            mode='same')/int(smooth_sec*sr_image)

    nevents, nframes = raster.shape
    buffer = np.floor(nframes / 2 / sr_image)

    sns.heatmap(raster, ax=ax, cbar=False, cmap=cmap, **kwargs)  # plot raster
    ax.plot(nevents - curve * nevents/y2scale - y2zero, 'r-')  # plot tuning curve
    ax.axvline(nframes / 2, color='g')  # plot event time
    if bs_rate is not None:
        ax.axhline(nevents - bs_rate / sr_image * nevents/y2scale - y2zero, color='g', linestyle='--')  # plot baseline rate
    ax.set_title('Cell ' + str(cell_id))

    ax.set_xticks([0, nframes / 2, nframes])
    if labelx:  # Label bottom row
        ax.set_xticklabels([str(-buffer), '0', str(buffer)])
        ax.set_xlabel('Time from ' + events + '(s)')

    if np.any(sig_bins):  # add a star/dot over all bins with significant tuning
        curve_plot = nevents - curve * nevents/y2scale - y2zero
        # ax.plot(sig_bins, curve_plot[sig_bins] - 5, 'r*')
        ax.plot(sig_bins, np.ones_like(sig_bins), sig_style)

    ax.set_yticks([0.5, nevents - 0.5])
    if labely:  # Label left side
        ax.set_yticklabels(['0', str(nevents)])
        ax.set_ylabel(events + ' #')

    secax = None
    if labely2:  # Add second axis and label
        secax = ax.secondary_yaxis('right', functions=(lambda y1: y2scale * (nevents - y1 - y2zero) / nevents,
                                                       lambda y: nevents * (1 - y / y2scale) - y2zero))
        secax.set_yticks([0, (nevents - y2zero) / nevents * y2scale])
        secax.tick_params(labelcolor='r')
        secax.set_ylabel(r'$p_{event}$', color='r')

    sns.despine(ax=ax)

    return ax, secax


def plot_speed_activity_xcorr(activity, speed, time_buffer_sec, sr, axcorr, label, labelx=True):
    """Plot cross-correlation between neural activity and speed. Designed for ensemble activity, probably
    will look wonky for single cells."""

    maxlags = np.round(time_buffer_sec * sr).astype('int')
    axcorr.xcorr(activity, speed, maxlags=maxlags)
    axcorr.set_xticks([-maxlags, 0, maxlags])
    axcorr.axvline(0, color='k')
    axcorr.set_xticklabels([str(-time_buffer_sec), "0", str(time_buffer_sec)])
    axcorr.set_title(label)
    if labelx:
        axcorr.set_xlabel('Lag (s)')
    axcorr.set_ylabel('Cross-Corr.')

    axcorr.autoscale(enable=True, axis='x', tight=True)  # Make it plot to very ends
    sns.despine(ax=axcorr)


def plot_PSA_w_freezing(mouse, arena, day, sort_by='first_event', day2=False, ax=None, inactive_cells='black',
                        plot_corr=False, **kwargs):
    """Plot *raw* calcium event rasters across whole session with velocity trace overlaid in red and freezing epochs
    overlaid in green. Can sort by various interesting metrics and apply that across days.
    :param mouse: str
    :param arena: 'Open' or 'Shock'
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param sort_by: how to sort neurons, options are: 'first_event', 'move_event_rate', 'freeze_event_rate', 'MMI', or None
    :param day2: 2nd day to use sorted by 1st day sorting (new neurons at the end). False (default) = plot one day only.
    :param ax: axes to plot into, default (None) = create new figure and axes.
    :param inactive_cells: str, 'black' (default) or 'white' plots inactive neuron rows all that color,
    'ignore' = remove rows altogether, keeping only cells active in both sessions
    :param plot_corr: bool, plots correlation between sort metrics across days , default = False
    :param kwargs: freezing related parameters for calculating freezing with all 'sort_by' options except 'first_event'.
    See er_plot_functions.detect_freezing(). Can also toggle 'batch_map_use' to True or False for sorting across days.
    :return: fig: main figure plot, if day2 == True, also returns fig handle for correlation scatterplot
    """

    # Sub-function to parse out PSA and velocity/freezing data
    def getPSA_and_freezing(mouse, arena, day, **kwargs):
        dir_use = erp.get_dir(mouse, arena, day)
        PF = pf.load_pf(mouse, arena, day)

        video_t = erp.get_timestamps(str(dir_use))
        video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

        # convert freezing indices to timestamps
        freezing, velocity = erp.detect_freezing(str(dir_use), arena=arena, **kwargs)
        freezing_epochs, freezing_times = get_freezing_times(mouse, arena, day)

        # get boolean of freezing indices in neural data
        freeze_bool = align_freezing_to_PSA(PF.PSAbool_align, PF.sr_image, freezing, video_t)

        # Now get time for imaging
        t_imaging = np.arange(0, PF.PSAbool_align.shape[1]) / PF.sr_image

        return PF.PSAbool_align, PF.sr_image, video_t, velocity, freezing_times, freeze_bool

    # Plot sub-function
    def plotPSAoverlay(PSAuse, sr_image, video_t, velocity, freezing_times, mouse, arena, day, ax,
                       ignore_neuron_bool=None, inactive_color='black'):
        # NRK todo: if plotting by MMI or event rate, overlay those metrics on the y-axis to see how well they line up across days!!!
        # Basically you need to a) update gridspec to have either a 1x5 or 1x10 grid, 0:4 = heatmap, 4=yplot, 5:10=heatmap,
        # 10 = plot, then you need to update MMI or event rate in the same manner you update PSAbool below.
        
        SFvel = 4  # Factor to scale velocity by for overlay below

        # Keep only specified rows if applicable
        if ignore_neuron_bool is not None:
            PSAuse = PSAuse[np.bitwise_not(ignore_neuron_bool)]
        else:
            # Make neuron rasters black if specified
            if inactive_color == 'black':
                PSAuse[np.isnan(PSAuse)] = 0
                
        sns.heatmap(data=PSAuse, ax=ax, xticklabels=1000, yticklabels=50, cbar=False)
        nneurons = PSAuse.shape[0]
        ax.plot(video_t * sr_image, velocity * -SFvel + nneurons / 2, color=[1, 0, 0, 0.5], linewidth=1)

        for freeze_time in freezing_times:
            ax.axvspan(freeze_time[0] * sr_image, freeze_time[1] * sr_image, color=[0, 1, 0, 0.4])

        # Pretty things up and label
        ax.tick_params(axis='y', rotation=0)
        ax.set_xticklabels([int(int(label.get_text()) / sr_image) for label in ax.get_xticklabels()])
        title_append = ': coactive cells only' if ignore_neuron_bool is not None else ': all cells'
        ax.set_title(mouse + ' ' + arena + ' Day ' + str(day) + title_append)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron # sorted by ' + str(sort_by))

    # Sorting sub-function
    def sort_PSA(mouse, arena, day, PSAbool_in, freeze_bool, sort_by, **kwargs):
        """Sort PSA according in `sort_by` metric"""

        # Next, sort PSAbool appropriately
        nneurons = PSAbool_in.shape[0]
        if sort_by is None:
            sort_ind, sort_array = np.arange(0, nneurons), None
        elif sort_by == 'first_event':  # Sort by first calcium event time
            PSAuse, sort_ind = helpers.sortPSA(PSAbool_in)
            sort_array = None
        elif sort_by == 'move_event_rate':
            sort_array = move_event_rate(PSAbool_in, freeze_bool)
        elif sort_by == 'freeze_event_rate':
            sort_array = freeze_event_rate(PSAbool_in, freeze_bool)
        elif sort_by == 'MMI':
            sort_array = motion_modulation_index(mouse, arena, day, **kwargs)

        if sort_by != 'first_event':
            PSAuse, sort_ind = helpers.sortPSA(PSAbool_in, sort_by=sort_array)

        return PSAuse, sort_ind, sort_array

    def ploty_sort_metric(sort_metric, axmetric, axpsa, sort_metric_name):
        """Plots metric by which PSAbool is sorted next to PSAbool on the y-axis. inputs are the (already sorted) metric
        by which cells are sorted, axes to plot into, PSAbool axes, and metric name"""
        axmetric.plot(sort_metric, range(len(sort_metric)), '.')
        axmetric.invert_yaxis()
        axmetric.set_xlabel(sort_metric_name)
        axmetric.axes.yaxis.set_visible(False)
        axmetric.set_ylim(axpsa.get_ylim())
        sns.despine(ax=axmetric)

    # First get PSA and velocity data
    PSAbool_align, sr_image, video_t, velocity, freezing_times, freeze_bool = \
        getPSA_and_freezing(mouse, arena, day, **kwargs)
    nneurons1 = PSAbool_align.shape[0]

    # Now sort
    PSAuse, sort_ind, sort_array = sort_PSA(mouse, arena, day, PSAbool_align, freeze_bool, sort_by, **kwargs)

    # Now set up plotting
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        nplots = 12 if day2 else 6
        gs = gridspec.GridSpec(nrows=1, ncols=nplots, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:4]) if sort_by is not None else fig.add_subplot(gs[0, 0:6])
        ax1sort_met = fig.add_subplot(gs[0, 4]) if sort_by is not None else None

    # append a block of NaN rows with 0th dimension = # new neurons if plotting across days
    if day2:
        # register and apply sorting to sort_array from above.
        neuron_map = pfs.get_neuronmap(mouse, arena, day, arena, day2, **kwargs)
        reg_session = sd.find_eraser_session(mouse, arena, day2)
        good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)

        nframes = PSAuse.shape[1]

        PSAuse = np.concatenate((PSAuse, np.ones((len(new_ind), nframes))*np.nan))

    # Now plot 2nd day if looking across days! Do this first to identify cells active across both days.
    if day2:
        PSAbool_align2, sr_image2, video_t2, velocity2, freezing_times2, freeze_bool2 = \
            getPSA_and_freezing(mouse, arena, day2, **kwargs)

        # next register and apply sorting to sort_ind from above.
        neuron_map = pfs.get_neuronmap(mouse, arena, day, arena, day2, **kwargs)
        reg_session = sd.find_eraser_session(mouse, arena, day2)
        good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)

        # now sort 2nd session cells by original session order
        sort_ind_reg = neuron_map[sort_ind]

        # Now sort new cells by same metric and append cell ids to sort_ind_reg
        PSAuse2_reg, sort_ind2, sort_array2 = sort_PSA(mouse, arena, day2, PSAbool_align2, freeze_bool2, sort_by,
                                                       **kwargs)

        PSAreg, sort_array2reg = [], []
        nframes_reg = PSAuse2_reg.shape[1]
        for ind in sort_ind_reg:
            if not np.isnan(ind):  # Add actual cell activity
                psa_to_add = PSAbool_align2[int(ind)]
                sort_met_add = sort_array2[int(ind)]
            else:  # Add in all nans if not active the next day
                psa_to_add = np.ones(nframes_reg)*np.nan
                sort_met_add = np.nan
            PSAreg.append(psa_to_add)  # append psa
            sort_array2reg.append(sort_met_add)
        PSAreg = np.asarray(PSAreg)  # convert from list to array
        sort_array2reg = np.asarray(sort_array2reg)

        # Now add in new cells at bottom
        PSAreg = np.concatenate((PSAreg, PSAuse2_reg[new_ind]), axis=0)
        sort_array2reg = np.concatenate((sort_array2reg, np.ones(len(new_ind))*np.nan), axis=0)

        # find out cells that are inactive in one of the sessions
        if inactive_cells == 'ignore':
            inactive_bool = np.bitwise_or(np.all(np.isnan(PSAreg), axis=1), np.all(np.isnan(PSAuse), axis=1))
        else:
            inactive_bool = np.ones(PSAuse.shape[0], dtype='bool')

        # Plot reg session
        ax2 = fig.add_subplot(gs[0, 6:10]) if sort_by is not None else fig.add_subplot(gs[0, 6:11])
        ax2sort_met = fig.add_subplot(gs[0, 10]) if sort_by is not None else None
        plotPSAoverlay(PSAreg, sr_image2, video_t2, velocity2, freezing_times2, mouse, arena, day2, ax=ax2,
                       ignore_neuron_bool=inactive_bool, inactive_color=inactive_cells)
        ax2.axhline(nneurons1 + 0.5, 0, 1, color='r', linestyle='--')

        # Plot sort metric next to raw data
        if sort_by is not None:
            sort_metric2_good = sort_array2reg[np.bitwise_not(inactive_bool)]
            ploty_sort_metric(sort_metric2_good, ax2sort_met, ax2, sort_by)

    # Finally plot first session
    plotPSAoverlay(PSAuse, sr_image, video_t, velocity, freezing_times, mouse, arena, day, ax=ax1,
                   ignore_neuron_bool=inactive_bool, inactive_color=inactive_cells)
    if sort_by is not None:  # plot sort metric on y-axis
        sort_metric_good = sort_array[sort_ind][np.where(np.bitwise_not(inactive_bool))[0]]
        ploty_sort_metric(sort_metric_good, ax1sort_met, ax1, sort_by)

    if day2:
        ax1.axhline(nneurons1 + 0.5, 0, 1, color='r', linestyle='--')
        ax2.set_ylabel('Sorted by Day ' + str(day))
        if plot_corr:
            figb, axb = plt.subplots()
            axb.plot(sort_metric_good, sort_metric2_good, '.')
            axb.set_xlabel(sort_by + ' Day ' + str(day))
            axb.set_ylabel(sort_by + ' Day ' + str(day2))
            axb.set_title(mouse + ': ' + arena)

            # calculate and plot correlation
            r, p = stats.spearmanr(sort_metric_good, sort_metric2_good, nan_policy='omit')
            xlims_use = np.asarray([-1, 1]) if sort_by == 'MMI' else np.asarray(axb.get_xlim())
            axb.plot(xlims_use, xlims_use*r, 'r-')
            if sort_by == "MMI":
                axb.text(0.375, -0.5, 'r = ' + f"{r:0.3g}")
                axb.text(0.375, -0.625, 'p = ' + f"{p:0.3g}")
            else:
                ylims_use = np.asarray(axb.get_ylim())
                axb.text(0.375*xlims_use[1], 0.2*ylims_use[1], 'r = ' + f"{r:0.3g}")
                axb.text(0.375*xlims_use[1], 0.1*ylims_use[1], 'p = ' + f"{p:0.3g}")
            sns.despine(ax=axb)

    if not day2 or not plot_corr:
        return fig
    elif day2 and plot_corr:
        return fig, figb


# This should be obsolete or in covariance_analysis module
# def scatter_cov_across_days(cov_mat: np.ndarray, cells: np.ndarray or None = None,
#                             include_silent: bool = False, ax=None, xlabel='Base Day',
#                             ylabel='Reg Day', sig_thresh: float or None = None, plot: bool = True,
#                             label_by_state=False, **kwargs) -> plt.Axes:
#     """Plot covariance matrix across days.  Takes in specially formatter matrix where lower triangle = base day
#     covariance and upper triangle = reg day covariance.  0s across the entire row of the upper triangle = silent cells
#
#     **kwargs go to matplotlib.plot"""
#
#     base_cov, reg_cov = get_cov_pairs_from_mat(cov_mat, cells, include_silent)
#
#     # Label by state if designated.
#     if sig_thresh is not None:  # Keep only significant pairs of cells from day 1
#         sig_cov_bool = base_cov > sig_thresh
#     else:
#         sig_cov_bool = np.ones_like(base_cov, dtype=bool)
#
#     # Finally plot it
#     if plot:
#         if ax is None:
#             _, ax = plt.subplots()
#
#         if not label_by_state:
#             ax.plot(base_cov, reg_cov, '.', **kwargs)
#         else:
#             hu, = ax.plot(base_cov[np.bitwise_not(sig_cov_bool)], reg_cov[np.bitwise_not(sig_cov_bool)], 'r.')
#             base_conn, reg_conn = base_cov[sig_cov_bool], reg_cov[sig_cov_bool]
#             strengthened = np.greater(reg_conn, base_conn)
#             weakened = np.greater(base_conn, reg_conn)
#             hs, = ax.plot(base_conn[strengthened], reg_conn[strengthened], 'g.')
#             hw, = ax.plot(base_conn[weakened], reg_conn[weakened], 'b.')
#             ax.legend((hu, hs, hw), ('Unpaired', 'Strengthen', 'Weaken'))
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         lim_min = np.min((xlim[0], ylim[0]))
#         lim_max = np.max((xlim[1], ylim[1]))
#
#         ax.plot([-0.1, 1], [-0.1, 1], 'r--')
#         ax.set_xlim((lim_min, lim_max))
#         ax.set_ylim((lim_min, lim_max))
#
#         ax.set_xlabel(xlabel + ' Cov.')
#         ax.set_ylabel(ylabel + ' Cov.')
#         if sig_thresh is not None:
#             ax.set_title(f'> {sig_thresh} std pairs only')
#         sns.despine(ax=ax)
#
#         return ax, np.vstack((base_cov, reg_cov))
#
#     else:
#         return np.vstack((base_cov, reg_cov))
#

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')  # This is a bugfix to make sure plots don't always stay on top of ALL applications

    # ts = TuningStability('Shock', 'freeze_onset', 0.01)  # Load in tuningstability object
    #
    # base_day = 1
    # metric_plot = 'event_rates'
    # delta = True
    # fig, ax = ts.plot_metric_stability_by_group(base_day=base_day, metric_plot=metric_plot, delta=delta,
    #                                             size=2, alpha=0.7, jitter=0.15)
    MD2 = MotionTuning('Marble14', 'Shock', 2)
    MD2.gen_pe_rasters('freeze_onset')


    pass