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

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers
from helpers import contiguous_regions


class MotionTuning:
    def __init__(self, mouse, arena, day):
        self.session = {'mouse': mouse, 'arena': arena, 'day': day}

        # Get PSAbool and freezing info
        self.PSAbool, self.freeze_bool = get_freeze_bool(mouse, arena, day)
        self.event_rates = self.PSAbool.sum(axis=1)/self.PSAbool.shape[1]
        self.freezing_indices, self.freezing_times = get_freezing_times(mouse, arena, day)

        # Get initial estimate of motion-modulated vs freeze modulated cells - very rough
        # don't really use - calculate later through function if needs be!
        # print('calculating rough modulation')
        # self.p, self.ER = calc_sig_modulation(mouse, arena, day)
        # self.freeze_cells_rough = np.where(self.p['MMI'] > 0.95)[0]
        # self.move_cells_rough = np.where(self.p['MMI'] < 0.05)[0]

        # Get sample rate
        dir_use = pf.get_dir(mouse, arena, day)
        self.dir_use = Path(dir_use)
        im_data_file = path.join(dir_use, 'FinalOutput.mat')
        im_data = sio.loadmat(im_data_file)
        try:
            self.sr_image = im_data['SampleRate'].squeeze()
        except KeyError:
            self.sr_image = 20

        self.pe_rasters = {'freeze_onset': None, 'move_onset': None}
        self.perm_rasters = {'freeze_onset': None, 'move_onset': None}

        try:  # Load in previously calculated tunings
            self.load_sig_tuning()
        except FileNotFoundError:  # if not saved, initialize
            self.sig = {'freeze_onset': {}, 'move_onset': {}}

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
    def __init__(self, mouse: str, arena: str or list, days: list = [-1, 4, 1, 2], events: str = 'freeze_onset'):
        """
        Create class for tracking motion or freezing tuning of cells across days.
        :param mouse: str of form 'Marble##'
        :param arena: str in ['Open', 'Shock'] or list matching len(days)
        :param days: int in [-2, -1, 0, 4, 1, 2, 7] though day 0 generally not analyzed in 'Shock' due to short
        (60 sec) recording time
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
            self.motion_tuning[arena][day] = MotionTuning(mouse, arena, day)  # Get motion tuning for each day above.
            self.motion_tuning[arena][day].gen_pe_rasters(events=events)  # generate freeze_onset rasters by default
            self.rois[arena][day] = helpers.get_ROIs(mouse, arena, day)

        # Initialize map between sessions
        self.map = {'map': None, 'base_day': None, 'base_arena': None}

    def plot_raster_across_days(self, cell_id, base_arena='Shock', base_day=1,
                                labelx=True, ax=None, batch_map=True, plot_ROI=True, **kwargs):
        """Plots a cell's peri-event raster on base_day and tracks backward/forward to all other days in object.
        e.g. if you have a freezing cell emerge on day1 and want to see what it looked like right before/after,
        use base_day=1

        :param cell_id: int, cell to plot on base_day and track forward/backward to other days_plot
        :param base_arena: str in ['Open', 'Shock']
        :param base_day: int
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
        self.assemble_map(base_day=base_day, base_arena=base_arena)  # Create/get neuron map
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

                labely = True if idd == 0 or not ylabel_added else False  # label y if on left side
                labely2 = True if idd == last_good_neuron else False  # label y2 if on right side

                # plot rasters
                _, secax = plot_raster(raster_plot, cell_id=id_plot, bs_rate=bs_rate, events=events,
                                       labelx=labelx, labely=labely, labely2=labely2, ax=ax[idd])
                ax[idd].set_title(arena + ' Day ' + str(day) + ': Cell ' + str(id_plot))
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
        look at how well a cell maintains its freeze or motion related tuning.  Compare to chance?
        Not sure how to get - take peak from uniform distribution across all bins and calculate dist for
        each cell?
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
        pvals = np.ones(len(self.days))*np.nan

        window_half = smooth_window/2  # Get half of smoothing window

        # Loop through each day, compare to base_day, and get peak tuning location and pval
        tuning_curve_all = []
        for idd, (arena, day, id) in enumerate(zip(self.arenas, self.days, reg_id)):

            if id >= 0:  # only calculate if valid mapping between neurons
                tuning_curves = gen_motion_tuning_curve(self.motion_tuning[arena][day].pe_rasters[self.events])
                tuning_curve_all.append(tuning_curves[id])

                # Get max event rate and its location (time)
                locs[idd], event_rates[idd] = get_tuning_max(tuning_curves[id], window=smooth_window)

                # Grab pvalues for tuning curve
                p_use = self.motion_tuning[base_arena][day].sig[self.events]['pval'][id]

                # Calculate mean p-value along the curve at the location of the peak (within the smoothing window)
                pvals[idd] = np.mean(p_use[int(locs[idd]-window_half):int(locs[idd]+window_half)])

        # Now run correlation between all days
        corrs, pcorrs = np.ones_like(tuning_curve_all)*np.nan, np.ones_like(tuning_curve_all)*np.nan
        for idd, (curve, id) in enumerate(zip(tuning_curve_all, reg_id)):

            if id >= 0:  # only get legit mapping values
                corrs[idd], pcorrs[idd], _ = pfs.spearmanr_nan(curve, tuning_curve_all[base_id])

        corr = {'corrs': corrs, 'pvals': pcorrs}

        return locs, event_rates, pvals, corr


def get_freezing_times(mouse, arena, day, **kwargs):
    """Identify chunks of frames and timestamps during which the mouse was freezing

    :param mouse: str
    :param arena: 'Open' or 'Shock'
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param kwargs: Freezing parameters to use. See er_plot_functions.detect_freezing()
    :return: freezing_epochs: list of start and end indices of each freezing epoch in behavioral video
             freezing_times: list of start and end times of each freezing epoch
    """
    dir_use = erp.get_dir(mouse, arena, day)

    video_t = erp.get_timestamps(str(dir_use))
    freezing, velocity = erp.detect_freezing(str(dir_use), arena=arena, **kwargs)
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # convert freezing indices to timestamps
    freezing_indices = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_indices]

    return np.asarray(freezing_indices), np.asarray(freezing_times)


def align_freezing_to_PSA(PSAbool, sr_image, freezing, video_t):
    """
    Align freezing times to neural data.
    :param PSAbool: nneurons x nframes_imaging boolean ndarray of putative spiking activity
    :param sr_image: frames/sec (int)
    :param freezing: output of er_plot_functions.detect_freezing() function.
    :param video_t: video frame timestamps, same shape as `freezing`
    :return: freeze_bool: boolean ndarray of shape (nframes_imaging,) indicating frames where animals was freezing.
    """

    # First get imaging parameters and freezing in behavioral video timestamps
    nneurons, nframes = PSAbool.shape
    freezing_epochs = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_epochs]

    # Set up boolean to match neural data shape
    freeze_bool = np.zeros(nframes, dtype='bool')
    PSAtime = np.arange(0, nframes)/sr_image

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
    # First get directory and neural data
    dir_use = erp.get_dir(mouse, arena, day)
    PF = pf.load_pf(mouse, arena, day)

    # Now get behavioral timestamps and freezing times
    video_t = erp.get_timestamps(str(dir_use))
    freezing, velocity = erp.detect_freezing(str(dir_use), arena=arena, **kwargs)
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # Now align freezing to neural data!
    freeze_bool = align_freezing_to_PSA(PF.PSAbool_align, PF.sr_image, freezing, video_t)

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


def get_PE_raster(psa, event_starts, buffer_sec=[2, 2], sr_image=20):
    """ Gets peri-event rasters for +/-buffers sec from all event start times in event_starts
    :param psa: activity for one cell at sr_image
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


def plot_raster(raster, cell_id=None, sig_bins=None, bs_rate=None, y2scale=0.2, events='trial',
                labelx=True, labely=True, labely2=True, sr_image=20, ax=None):
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
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches([2.5, 3])

    curve = gen_motion_tuning_curve(raster).squeeze()

    nevents, nframes = raster.shape
    buffer = np.floor(nframes / 2 / sr_image)

    sns.heatmap(raster, ax=ax, cbar=False)
    ax.plot(nevents - curve * nevents * 4, 'r-')
    ax.axvline(nframes / 2, color='g')
    if bs_rate is not None:
        ax.axhline(nevents - bs_rate * nevents * 4, color='b', linestyle='--')
    ax.set_title('Cell ' + str(cell_id))
    if labelx:  # Label bottom row
        ax.set_xticks([0, nframes / 2, nframes])
        ax.set_xticklabels([str(-buffer), '0', str(buffer)])
        ax.set_xlabel('Time from ' + events + '(s)')

    if sig_bins:  # add a start over all bins with significant tuning
        pass

    if labely:  # Label left side
        ax.set_yticks([0.5, nevents - 0.5])
        ax.set_yticklabels(['0', str(nevents)])
        ax.set_ylabel(events + ' #')

    secax = None
    if labely2:  # Add second axis and label
        secax = ax.secondary_yaxis('right', functions=(lambda y1: y2scale * (nevents - y1) / nevents,
                                                       lambda y: nevents * (1 - y / y2scale)))
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
    mmd = MotionTuningMultiDay('Marble24', 'Shock', days=[-1, 4, 1, 2])
    mmd.plot_raster_across_days(40, days_plot=[-1, 4, 1, 2], batch_map=False)

    pass