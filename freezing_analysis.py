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
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers
from helpers import contiguous_regions
import eraser_reference as err


class MotionTuning:
    """Identify and plot freeze and motion related cells
    **kwargs: inputs for getting freezing in eraser_plot_functions.get_freezing.
    """
    def __init__(self, mouse, arena, day, **kwargs):
        self.session = {'mouse': mouse, 'arena': arena, 'day': day}

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
            self.load_sig_tuning()
        except FileNotFoundError:  # if not saved, initialize
            print('No tunings found for this session - run .get_tuning_sig() and .save_sig_tuning()')
            self.sig = {'freeze_onset': {}, 'move_onset': {}}

    def get_prop_tuned(self, events: str = 'freeze_onset', **kwargs):
        """
        Gets proportion of neurons that exhibit freeze or motion related tuning
        :param events: str, 'freeze_onset' (default) or 'move_onset'
        :param kwargs: inputs to get_sig_neurons() to classify freeze or motion related movement
        :return:
        """
        ntuned = self.get_sig_neurons(events=events, **kwargs).shape[0]
        ntotal = self.sig[events]['pval'].shape[0]

        return ntuned/ntotal

    def gen_pe_rasters(self, events='freeze_onset', buffer_sec=[2, 2]):
        """Generate the rasters for all cells and dump them into a dictionary"""
        # Get appropriate event times to use
        if events in ['freeze_onset', 'move_onset']:
            event_starts = self.select_events(events)

        pe_rasters = [get_PE_raster(psa, event_starts, buffer_sec=buffer_sec,
                                    sr_image=self.sr_image) for psa in self.PSAbool]

        pe_rasters = np.asarray(pe_rasters)
        self.pe_rasters[events] = pe_rasters

        return pe_rasters

    def gen_perm_rasters(self, events='freeze_onset', buffer_sec=[2, 2], nperm=1000):
        """Generate shuffled rasters and dump them into a dictionary"""
        # Get appropriate cells and event times to use
        event_starts = self.select_events(events)

        # Loop through each cell and get its chance level raster
        print('generating permuted rasters - may take up to 1 minute')
        perm_rasters = np.asarray([shuffle_raster(psa, event_starts, buffer_sec=buffer_sec,
                                       sr_image=self.sr_image, nperm=nperm) for psa in self.PSAbool]).swapaxes(0, 1)
        self.perm_rasters[events] = perm_rasters

        return perm_rasters

    def get_tuning_sig(self, events='freeze_onset', buffer_sec=[2, 2], nperm=1000):
        """This function will calculate significance values by comparing event-centered tuning curves to
        chance (calculated from circular permutation of neural activity).
        :param cells:
        :param events:
        :param buffer_sec:
        :return:
        """

        # Load in previous tuning
        sig_use = self.sig[events]

        calc_tuning = True
        # Check to see if appropriate tuning already run and stored and just use that, otherwise calculate from scratch.
        if 'nperm' in sig_use:
            if sig_use['nperm'] == nperm:
                calc_tuning = False
                pval = sig_use['pval']

        if calc_tuning:
            print('calculating significant tuning for nperm=' + str(nperm))
            # check if both regular and permuted raster are run already!
            pe_rasters, perm_rasters = self.check_rasters_run(events=events,
                                                              buffer_sec=buffer_sec,  nperm=nperm)

            # Now calculate tuning curves and get significance!
            pe_tuning = gen_motion_tuning_curve(pe_rasters)
            perm_tuning = np.asarray([gen_motion_tuning_curve(perm_raster) for perm_raster in perm_rasters])
            pval = (pe_tuning < perm_tuning).sum(axis=0) / nperm

            # Store in class
            self.sig[events]['pval'] = pval
            self.sig[events]['nperm'] = nperm

            # Save to disk to save time in future
            self.save_sig_tuning()

        return pval

    def get_sig_neurons(self, events='freeze_onset', buffer_sec=[2, 2], nperm=1000,
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

    def check_rasters_run(self, events='freeze_onset', buffer_sec=[2, 2],  nperm=1000):
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

        elif not isinstance(pe_rasters, np.ndarray):
            pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)
            if not isinstance(perm_rasters, np.ndarray):
                perm_rasters = self.gen_perm_rasters(events=events, buffer_sec=buffer_sec, nperm=nperm)
            else:
                pe_rasters, perm_rasters = self.check_rasters_run(events=events, buffer_sec=buffer_sec, nperm=nperm)

        return pe_rasters, perm_rasters

    def save_sig_tuning(self):
        """Saves any significant tuned neuron data"""
        with open(self.dir_use / "sig_motion_tuning.pkl", 'wb') as f:
            dump(self.sig, f)

    def load_sig_tuning(self):
        """Loads any previously calculated tunings"""
        with open(self.dir_use / "sig_motion_tuning.pkl", 'rb') as f:
            self.sig = load(f)

    def plot_pe_rasters(self, cells='freeze_fine', events='freeze_onset', buffer_sec=[2, 2], **kwargs):
        """ Plot rasters of cells at either movement or freezing onsets.

        :param cells: str to auto-select either cells that have significant tuning to move or freeze onset
        at a fine ('move_fine' or 'freeze_fine' calculated with buffer_sec of motion/freezing onset)
        or rough ('move_rough' or 'freeze_rough', tuning calculated across all freezing/motion timepoints) timescale.
        Can also be a list of other events
        :param events: 'move_onset' or 'freeze_onset'.
        :param buffer_sec: int or size 2, array like of time(s) +/- event to plot
        :param kwargs:
        :return:
        """

        # NRK todo: better yet, fold this into a wrapper function. User must specify cells to plot.
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
                         str(self.session['day']) + ': ' + cells + ' cells: plot ' + str(plot))

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
                 **kwargs):
        """
        Create class for tracking motion or freezing tuning of cells across days.
        :param mouse: str of form 'Marble##'
        :param arena: str in ['Open', 'Shock'] or list matching len(days)
        :param days: int in [-2, -1, 0, 4, 1, 2, 7] though day 0 generally not analyzed in 'Shock' due to short
        (60 sec) recording time
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
            self.motion_tuning[arena][day] = MotionTuning(mouse, arena, day, **kwargs)  # Get motion tuning for each day above.
            self.motion_tuning[arena][day].gen_pe_rasters(events=events)  # generate freeze_onset rasters by default
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

    def plot_raster_across_days(self, cell_id, base_arena='Shock', base_day=1, alpha=0.01,
                                labelx=True, ax=None, batch_map=True, plot_ROI=True, **kwargs):
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
        :param **kwargs: see MotionTuning.gen_pe_rasters
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

                labely = True if idd == 0 or not ylabel_added else False  # label y if on left side
                labely2 = True if idd == last_good_neuron else False  # label y2 if on right side

                # plot rasters
                _, secax = plot_raster(raster_plot, cell_id=id_plot, bs_rate=bs_rate, events=events,
                                       labelx=labelx, labely=labely, labely2=labely2, ax=ax[idd],
                                       sig_bins=sig_bins)
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

        fig.suptitle(self.mouse + ': Across Days')

        return ax

    def assemble_map(self, base_day: int in [-2, -1, 4, 1, 2, 7], base_arena: str in ['Open', 'Shock'],
                     batch_map: bool = False):
        """
        Assembles all neuron mappings from base day to other days in self.days
        :param base_day:
        :param base_arena:
        :param batch_map: bool, how to register neurons across day: via batch map (False) or direct mapping (True, default).
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
        """Determines the probability a motion-tuned cell on basd day retains that tuning on a different day """
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
        metric_labels = [r'$\Delta_t$', r'$\Delta{ER} (1/s)$', 'p at peak', r'$\rho$', r'$p_{\rho}$']
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
                                       group_by: str in ['Group', 'Exp Group'] = 'Group'):


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
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((12.9, 4.75))
        met_name = metric_plot if not delta else 'Delta' + metric_plot

        # Plot scatter
        sns.stripplot(x='Day', y=met_name, data=df, hue=group_by, dodge=True, order=days_plot,
                      palette=pal_use, ax=ax[0])

        # This is the only easy way I could figure out to NOT duplicate labels in the legend
        group_rows = df.loc[:, group_by].copy()  # This generates warnings about chained indexing for some reason
        group_rows_ = ["_" + row for row in df[group_by]]
        df.loc[:, group_by] = group_rows_

        # Plot overlaying bar graph
        sns.barplot(x='Day', y=met_name, data=df, hue=group_by, dodge=True, order=days_plot,
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

        # now get stats and print out
        ycoord = 0.9
        if group_by == 'Group':  # NRK todo: learn how to use itertools! This works but is super non-elegant!
            for day in np.asarray(days_plot)[[base_day != d for d in days_plot]]:
                learners = df[np.bitwise_and(df['Group'] == 'Learners', df['Day'] == day)]
                nonlearners = df[np.bitwise_and(df['Group'] == 'Nonlearners', df['Day'] == day)]
                ani = df[np.bitwise_and(df['Group'] == 'ANI', df['Day'] == day)]
                stat, pval = stats.ttest_ind(learners[met_name], nonlearners[met_name], nan_policy='omit')
                print(f'2sided t-test Learners v Nonlearners day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ax[1].text(0.1, ycoord, f'2sided t-test Learners v Nonlearners day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ycoord -= 0.1
                stat, pval = stats.ttest_ind(learners[met_name], ani[met_name], nan_policy='omit')
                print(f'2sided t-test Learners v ANI day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ax[1].text(0.1, ycoord,f'2sided t-test Learners v ANI day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ycoord -= 0.1
                stat, pval = stats.ttest_ind(nonlearners[met_name], ani[met_name], nan_policy='omit')
                print(f'2sided t-test Learners v Nonlearners day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ax[1].text(0.1, ycoord, f'2sided t-test Nonlearners v ANI day {day}: pval= {pval:0.3g}, tstat={stat:0.3g}')
                ycoord -= 0.1

        else:
            print('Stats not yet enabled for "Exp Group" plotting')
            ax[1].text(0.1, 0.5, 'Stats not yet enabled for "Exp Group" plotting')

        ax[1].axis(False)

        return fig, ax

class TuningGeneralization:
    """Class to examine tuning curve generalization between arenas"""
    def __init__(self, events, alpha):
        pass


class DimReduction:
    """"Perform dimensionality reduction on cell activity to pull out coactive ensembles of cells"""
    def __init__(self, mouse: str, arena: str, day: int, bin_size=0.5, nICs=20):
        """Initialize ICA on data. Can also run PCA later if you want.

        :param mouse: str
        :param arena: str
        :param day: int
        :param bin_size: float, seconds, 0.5 = default
        :param nICs: int, 20 = default
        """

        self.mouse = mouse
        self.arena = arena
        self.day = day

        # Load in relevant data
        self.PF = pf.load_pf(mouse, arena, day)
        _, self.freeze_bool = get_freeze_bool(mouse, arena, day)
        self.freeze_ind = np.where(self.freeze_bool)[0]
        md = MotionTuning(mouse, arena, day)
        self.freeze_starts = md.select_events('freeze_onset')
        # Fix previously calculated occmap
        self.PF.occmap = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap)

        # First bin events
        PSAsmooth, PSAbin = [], []
        for psa in self.PF.PSAbool_align:
            PSAbin.append(bin_array(psa, int(bin_size * self.PF.sr_image)))  # Create non-overlapping bin array
        self.PSAbin = np.asarray(PSAbin)
        PSAbinz = stats.zscore(PSAbin, axis=1)  # Seems to give same results

        # Now calculate covariance matrix for all your cells using binned array
        self.cov_mat = np.cov(PSAbin)
        self.cov_matz = np.cov(PSAbinz)

        # Run ICA
        self._PCA_ICA(nICs=nICs)

    def _init_PCA_ICA(self, nPCs=50):
        """Initialize asssemblies based on PCA/ICA method from Lopes dos Santos (2013) and later
        van de Ven (2016) and Trouche (2016) """
        # Run PCA on binned PSA
        self.pca = PCA(self.cov_matz, nPCs)

        # Make into a dataframe for easy plotting
        self.pca.df = self.to_df(self.pca.covz_trans)

        # Calculated threshold for significant PCs
        q = self.PSAbinz.shape[1] / self.PSAbinz.shape[0]
        rho2 = 1
        self.pca.lambdamin = rho2 * np.square(1 - np.sqrt(1 / q))
        self.pca.lambdamax = rho2 * np.square(1 + np.sqrt(1 / q))

        # Identify assemblies whose eigenvalues exceed the threshold
        self.nA = np.max(np.where(self.pca.transformer.singular_values_ > self.pca.lambdamax)[0])
        assert self.nA < nPCs, '# assemblies = # PCs, re-run with a larger value for nPCs'

        # Create PCA-reduced activity matrix
        self.psign = self.transformer.components_[0:self.nA].T  # Grab top nA vector weights for neurons
        self.zproj = np.matmul(self.psign.T, self.PSAbin)  # Project neurons onto top nA PCAs
        self.cov_zproj = np.cov(self.zproj)  # Get covariance matrix of PCA-projected activity
        self.transformer = FastICA(random_state=0, whiten=False, max_iter=10000)  # Set up ICA
        self.w_mat = self.transformer.fit_transform(self.cov_zproj)  # Fit it

        # Get final weights of all neurons on PCA/ICA assemblies!
        self.v = np.matmul(self.psign, self.w_mat)

        # NRK todo: verify this works! Slightly different than other methods!
        # Dump into dataframe and get assembly activations over time
        self.df = self.to_df(self.v)
        self.activations = self.calc_activations('pcaica')

    def _init_ICA(self, nICs=20):
        """Perform ICA"""
        # Run ICA on the binned PSA
        self.ica = ICA(self.cov_matz, nICs)

        # Make into a dataframe for easy plotting
        self.ica.df = self.to_df(self.cov_trans)

        # Get activations for each IC over time
        self.ica.activations = self.calc_activations('ica')
        self.ica.nA = nICs

    def _init_PCA(self, nPCs=50):
        """Perform PCA. Will overwrite ICA results"""
        # Run PCA on binned PSA
        self.pca = PCA(self.cov_matz, nPCs)

        # Make into a dataframe for easy plotting
        self.pca.df = self.to_df(self.pca.cov_trans)

        # Get activations of each PC
        self.pca.activations = self.calc_activations('pca')
        self.pca.nA = nPCs

    @staticmethod
    def get_titles(dr_type: str in ['pcaica', 'pca', 'ica']):
        """Get plotting titles for different DR types - not sure if this is really used!"""
        if dr_type == 'pcaica':
            dim_red_type = 'PCA/ICA'
            dim_prefix = 'Assembly'
        elif dr_type == 'pca':
            dim_red_type = 'PCA'
            dim_prefix = 'PC'
        elif dr_type == 'ica':
            dim_red_type = 'ICA'
            dim_prefix = 'IC'

        return dim_red_type, dim_prefix

    def plot_assembly_fields(self, assembly: int, ncells_plot: int = 7,
                             dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica',
                             tmap_type: str in ['sm', 'us'] = 'sm', isrunning: bool = False):

        assert assembly < self.get_nA(dr_type), 'assembly must be less than # of ICs or PCs'

        # Identify top ncells_plot cells that most contribute to the assembly
        df = self.get_df(dr_type)
        assemb_wts = df.iloc[:, assembly + 1]
        top_cells = np.argsort(np.abs(assemb_wts))[-ncells_plot:][::-1]

        # Set up plots
        nplots = ncells_plot + 1
        nrows = np.ceil(nplots / 4).astype(int)
        fig, ax = plt.subplots(nrows, 4)
        fig.set_size_inches((nrows*5, 4.75))

        # Plot assembly spatial tuning first
        assemb_pf = self.make_assembly_field(assembly, dr_type, isrunning=isrunning)
        # pf.imshow_nan(assemb_pf, ax=ax[0][0])
        ax[0][0].imshow(assemb_pf)

        _, dim_prefix = self.get_titles(dr_type)
        ax[0][0].set_title(dim_prefix + ' #' + str(assembly))
        if tmap_type == 'sm':
            tmap_use = self.PF.tmap_sm
        elif tmap_type == 'us':
            tmap_use = self.PF.tmap_us

        # Plot each cell's tuning also
        for a, cell in zip(ax.reshape(-1)[1:nplots], top_cells):
            # pf.imshow_nan(self.PF.tmap_sm[cell], ax=a)
            a.imshow(tmap_use[cell])
            a.set_title(f'Cell {cell} wt: {assemb_wts[cell]:0.3g}')
        sns.despine(fig=fig, left=True, bottom=True)

    def make_occ_map(self, type: str in ['run', 'occ', 'immobile', 'freeze']):
        """Grab specific types of occupancy maps quickly and easily"""

        if type == 'run':
            occ_map_nan = self.PF.runoccmap.copy()
        elif type == 'occ':
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap)
        elif type == 'immobile':  # grab all immobility times
            good_bool = np.bitwise_not(self.PF.isrunning)
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap, good_bool=good_bool)
        elif type == 'freeze':  # grab only extended immobility times that count as actual freezing events
            good_bool = self.freeze_bool
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap, good_bool=good_bool)

        return occ_map_nan

    def make_assembly_field(self, assembly: int, dr_type: str in ['pcaica', 'pca', 'ica'], isrunning: bool = False):

        # Grab occ map to get correct size and shape for 2-d assembly field
        occ_map_nan = self.PF.runoccmap.copy()
        occ_map_nan[occ_map_nan == 0] = np.nan

        # Initialize maps - keep occ_map_check in case you change code later and need a sanity check!
        assemb_pf = np.zeros_like(np.rot90(self.PF.occmap, -1))
        occ_map_check = np.zeros_like(np.rot90(self.PF.occmap, -1))
        nx, ny = assemb_pf.shape

        # Get appropriate activations!
        activations = self.get_activations(dr_type)

        # Loop through each bin and add up assembly values in that bin
        for xb in range(nx):
            for yb in range(ny):
                in_bin_bool = np.bitwise_and((self.PF.xBin - 1) == xb, (self.PF.yBin - 1) == yb)
                if isrunning:
                    in_bin_bool = np.bitwise_and(in_bin_bool, self.PF.isrunning)
                assemb_pf[xb, yb] = activations[assembly][in_bin_bool].sum()
                occ_map_check[xb, yb] = in_bin_bool.sum()

        assemb_pf = np.rot90(assemb_pf, 1)
        occ_map_check = np.rot90(occ_map_check, 1)
        occ_map_check[occ_map_check == 0] = np.nan

        assemb_pf = assemb_pf * self.PF.occmap / np.nansum(self.PF.occmap.reshape(-1))
        assemb_pf[np.bitwise_or(self.PF.occmap == 0, np.isnan(self.PF.occmap))] = np.nan

        return assemb_pf

    def plot_rasters(self, dr_type: str in ['pcaica', 'pca', 'ica'], buffer: int = 6):
        """Plot rasters of assembly activation in relation to freezing"""

        # Get appropriate activations
        activations = self.get_activations(dr_type)

        ncomps = activations.shape[0]  # Get # assemblies
        ncols = 5
        nrows = np.ceil(ncomps / ncols).astype(int)
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_size_inches([3 * nrows, 15])

        ntrials = self.freeze_starts.shape[0]
        for ida, (a, act) in enumerate(zip(ax.reshape(-1), activations)):
            # Assemble raster for assembly and calculate mean activation
            assemb_raster = get_PE_raster(act, event_starts=self.freeze_starts, buffer_sec=[buffer, buffer],
                                          sr_image=self.PF.sr_image)
            act_mean = act.mean()
            # Hack to switch up signs - necessary?
            if act_mean < 0:
                assemb_raster = assemb_raster * -1
                act_mean = act_mean * -1
                suffix = 'f'
            else:
                suffix = ''

            plot_raster(assemb_raster, cell_id=ida, bs_rate=act_mean, sr_image=self.PF.sr_image, ax=a,
                        y2scale=0.35, y2zero=ntrials / 5, cmap='rocket')
            a.set_title(f'ICA {ida}{suffix}')

        return ax

    def plot_PSA_w_activation(self, comp_plot: int, dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica'):
        """Plots all calcium activity with activation of a particular assembly overlaid"""

        # Get appropriate activations
        activations = self.get_activations(dr_type)

        # First grab activity of a given assembly
        activation = activations[comp_plot]

        # Next, sort PSA according to that assembly
        df = self.get_df(dr_type)
        isort = df[comp_plot].argsort()
        # PSAsort_bin = self.PSAbin[isort[::-1]]
        PSAsort = self.PF.PSAbool_align[isort[::-1]]

        # Now, set up the plot
        fig, ax = plt.subplots()
        fig.set_size_inches([6, 3.1])
        sns.heatmap(PSAsort, ax=ax)
        _, dim_prefix = self.get_titles(dr_type)
        ax.set_title(f'{dim_prefix} # {comp_plot}')
        # sns.heatmap(PSAsort_bin, ax=ax[1])
        # ax[1].set_title('PSAbin ICA = ' + str(ica_plot))

        # plot freezing times
        ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind), 'r.')
        ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind) * (self.PF.PSAbool_align.shape[0] / 2 - 20), 'r.')
        ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind) * (self.PF.PSAbool_align.shape[0] - 5), 'r.')
        ax.plot(-activation * 100 + self.PF.PSAbool_align.shape[0] / 2, 'g-')
        ax.plot(-self.PF.speed_sm*10 + self.PF.PSAbool_align.shape[0]/2, 'b-')

        # Label things
        ax.set_ylabel('Sorted cell #')
        ax.set_xticklabels([f'{tick/self.PF.sr_image:0.0f}' for tick in ax.get_xticks()])
        ax.set_xlabel('Time (s)')

        return ax

    def calc_speed_corr(self, plot=True, dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica'):
        """Correlate activation of each component with speed or project onto freezing to visualize any freeze-related
        assemblies"""

        # Get appropriate activations
        activations = self.get_activations(dr_type)

        # Now correlate with speed and multiply by freezing
        freeze_proj, speed_corr = [], []
        for act in activations:
            act = np.asarray(act, dtype=float)
            corr_mat = np.corrcoef(np.stack((act, self.PF.speed_sm), axis=1).T)
            speed_corr.append(corr_mat[0, 1])
            freeze_proj.append((act * self.freeze_bool).sum())

        freeze_proj = np.asarray(freeze_proj)
        speed_corr = np.asarray(speed_corr)

        # Plot
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(speed_corr, freeze_proj)
            ax.set_xlabel('speed correlation')
            ax.set_ylabel('freeze PSAbool projection')
            for idc, (x, y) in enumerate(zip(speed_corr, freeze_proj)):
                ax.text(x, y, str(idc))
            dim_red_type, _ = self.get_titles(dr_type)
            ax.set_title(dim_red_type)
            sns.despine(ax=ax)

        return speed_corr, freeze_proj

    def activation_v_speed(self, dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica'):
        """Scatterplot of all components versus speed"""

        # Get appropriate activations
        activations = self.get_activations(dr_type)

        figc, axc = plt.subplots(4, 5)
        figc.set_size_inches([18, 12])
        dim_red_type, dim_prefix = self.get_titles(dr_type)
        for ida, (act, a) in enumerate(zip(activations, axc.reshape(-1))):
            a.scatter(self.PF.speed_sm, act, s=1)
            a.set_title(f'{dim_prefix} #{ida}')
            a.set_xlabel('Speed (cm/s)')
            a.set_ylabel(dim_red_type + ' activation')
        sns.despine(fig=figc)

    @staticmethod
    def to_df(cov_trans):
        """Transforms a reduced covariance matrix array to a dataframe"""
        df = pd.DataFrame(cov_trans)  # Convert to df

        # Add in cell # column - helpful for some plots later
        df["cell"] = [str(_) for _ in np.arange(0, df.shape[0])]
        cols = df.columns.to_list()

        # put cell # in first column
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        return df

    def calc_activations(self, dr_type: str in ['pcaica', 'pca', 'ica']):
        """Pull out activation of each component (IC or PC) over the course of the recording session"""

        # Get appropriate dataframe
        if dr_type == 'pcaica':
            df = self.df
        else:
            df = getattr(self, dr_type).df

        # Create array of activations
        activation = []
        for ica_weight in df.iloc[:, df.columns != 'cell'].values.swapaxes(0, 1):
            activation.append(np.matmul(ica_weight, self.PSAbool_align))
        act_array = np.asarray(activation)

        return act_array

    def get_activations(self, dr_type: str in ['pcaica', 'pca', 'ica']):

        if dr_type == 'pcaica':
            activations = self.activations
        else:
            activations = getattr(self, dr_type).activations

        return activations

    def get_nA(self, dr_type: str in ['pcaica', 'pca', 'ica']):

        if dr_type == 'pcaica':
            nA = self.nA
        else:
            nA = getattr(self, dr_type).nA

        return nA

    def get_df(self, dr_type: str in ['pcaica', 'pca', 'ica']):

        if dr_type == 'pcaica':
            df = self.df
        else:
            df = getattr(self, dr_type).df

        return df


class PCA:
    def __init__(self, cov_matz: np.ndarray, nPCs: int = 50):
        """Perform PCA on z-scored covariance matrix"""
        # Run PCA on binned PSA
        self.transformer = PCA(n_components=nPCs)
        self.cov_trans = self.transformer.fit_transform(cov_matz)


class ICA:
    def __init__(self, cov_matz: np.ndarray, nICs: int = 20):
        """Perform ICA on z-scored covariance matrix"""
        # Run ICA on the binned PSA
        self.transformer = FastICA(n_components=nICs, random_state=0)
        self.cov_trans = self.transformer.fit_transform(cov_matz)


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
        :param freezingPSA: boolean ndarray of shape (nframes_imaging,) indicating frames where animals was freezing.
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


def get_PE_raster(psa, event_starts, buffer_sec=[2, 2], sr_image=20):
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

    pe_raster = np.asarray(raster_list[1:])

    return pe_raster


def shuffle_raster(psa, event_starts, buffer_sec=[2, 2], sr_image=20, nperm=1000):
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
    if (npts := (max_val == curve_smooth).sum()) > 1:
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


def plot_raster(raster, cell_id=None, sig_bins=None, bs_rate=None, y2scale=0.25, events='trial',
                labelx=True, labely=True, labely2=True, sr_image=20, ax=None, y2zero=0, cmap='rocket'):
    #NRK todo: change bs_rate plot to incorporate sample rate. currently too high!!!
    """Plot peri-event raster with tuning curve overlaid.

    :param raster: nevents x nframes array
    :param cell_id: int, cell # to label with
    :param sig_bins: bool, frames with significant tuning to label with *s
    :param bs_rate: float, baseline rate outside of events
    :param y2scale: float, 0.2 = default, scale for plotting second y-axis (event rate)
    :param events: str, for x and y labels
    :param labelx: bool
    :param labely: bool
    :param labely2: bool
    :param sr_image: int, default = 20 fps
    :param ax: ax to plot into, default = None -> create new fig
    :param y2zero: location of y2 axis zero point in # trials
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([2.5, 3])

    curve = gen_motion_tuning_curve(raster).squeeze()

    nevents, nframes = raster.shape
    buffer = np.floor(nframes / 2 / sr_image)

    sns.heatmap(raster, ax=ax, cbar=False, cmap=cmap)  # plot raster
    ax.plot(nevents - curve * nevents/y2scale - y2zero, 'r-')  # plot tuning curve
    ax.axvline(nframes / 2, color='g')  # plot event time
    if bs_rate is not None:
        ax.axhline(nevents - bs_rate / sr_image * nevents/y2scale - y2zero, color='g', linestyle='--')  # plot baseline rate
    ax.set_title('Cell ' + str(cell_id))
    if labelx:  # Label bottom row
        ax.set_xticks([0, nframes / 2, nframes])
        ax.set_xticklabels([str(-buffer), '0', str(buffer)])
        ax.set_xlabel('Time from ' + events + '(s)')

    if np.any(sig_bins):  # add a star/dot over all bins with significant tuning
        curve_plot = nevents - curve * nevents/y2scale - y2zero
        # ax.plot(sig_bins, curve_plot[sig_bins] - 5, 'r*')
        ax.plot(sig_bins, np.ones_like(sig_bins), 'r.')

    if labely:  # Label left side
            ax.set_yticks([0.5, nevents - 0.5])
            ax.set_yticklabels(['0', str(nevents)])
            ax.set_ylabel(events + ' #')

    secax = None
    if labely2:  # Add second axis and label
        secax = ax.secondary_yaxis('right', functions=(lambda y1: y2scale * (nevents - y1 - y2zero) / nevents,
                                                       lambda y: nevents * (1 - y / y2scale) - y2zero))
        secax.set_yticks([0, y2scale])
        secax.tick_params(labelcolor='r')
        secax.set_ylabel(r'$p_{event}$', color='r')

    sns.despine(ax=ax)

    return ax, secax


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


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    DR = DimReduction('Marble07', 'Open', -2)
    DR.plot_assembly_fields(0)

    pass