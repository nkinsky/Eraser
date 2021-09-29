import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from os import path
import scipy.io as sio

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers


class Freeze:
    def __init__(self, mouse, arena, day, p_thresh=0.05):
        self.session = {'mouse': mouse, 'arena': arena, 'day': day}

        # Get PSAbool and freezing info
        self.PSAbool, self.freeze_bool = get_freeze_bool(mouse, arena, day)
        self.freezing_indices, self.freezing_times = get_freezing_times(mouse, arena, day)

        # Get initial estimate of motion-modulated vs freeze modulated cells
        self.p, self.ER = calc_sig_modulation(mouse, arena, day)
        self.freeze_cells = np.where(self.p['MMI'] > 0.95)[0]
        self.move_cells = np.where(self.p['MMI'] < 0.05)[0]

        # Get sample rate
        dir_use = pf.get_dir(mouse, arena, day)
        im_data_file = path.join(dir_use, 'FinalOutput.mat')
        im_data = sio.loadmat(im_data_file)
        try:
            self.sr_image = im_data['SampleRate'].squeeze()
        except KeyError:
            self.sr_image = 20

        self.pe_rasters = {'freeze_cells': {'freeze_onset': {}, 'move_onset': {}},
                           'move_cells': {'freeze_onset': {}, 'move_onset': {}}}


    def gen_pe_rasters(self, cells='freeze', events='freeze_onset', buffer_sec=[2, 2]):
        """Generate the rasters and dump them into a dictionary"""
        # Get appropriate cells and event times to use
        cells_use, event_starts = self.select_events(cells, events)

        pe_rasters = [get_PE_raster(self.PSAbool[cell], event_starts, buffer_sec=buffer_sec,
                                    sr_image=self.sr_image) for cell in cells_use]

        self.pe_rasters[cells + '_cells'][events]['data'] = pe_rasters

        return pe_rasters

    def gen_perm_rasters(self, cells='freeze', events='freeze_onset', buffer_sec=[2, 2]):
        """Generate shuffled rasters and dump them into a dictionary"""
        # Get appropriate cells and event times to use
        cells_use, event_starts = self.select_events(cells, events)

        perm_rasters = [shuffle_raster(self.PSAbool[cell], event_starts, buffer_sec=buffer_sec,
                                       sr_image=self.sr_image) for cell in cells_use]
        self.pe_rasters[cells + '_cells'][events]['perm'] = perm_rasters

        return perm_rasters

    def gen_tuning_curves(self):
        """This function will generate tuning curves and p-values for each time bin based on real and shuffled data"""
        pass

    def plot_pe_rasters(self, cells, events):
        raster_use = self.pe_rasters[cells + '_cells'][events]['data']
        if cells == 'freeze':
            cell_ids = self.freeze_cells
        elif cells == 'move':
            cell_ids = self.move_cells

        # NRK todo: add in tuning curves based on permutations if calculated with significance indicated
        # perm_use = self.pe_rasters[cells + '_cells'][events]['perm']

        # hopefully future proof for rasters as either a list (as developed)
        ncells = len(raster_use) if type(raster_use) == list else raster_use.shape[0]
        nframes = raster_use[0].shape[1]

        nplots = np.ceil(ncells/25)
        for plot in range(nplots):
            fig, ax = plt.subplots(5, 5)
            fig.set_size_inches([12, 6.9])
            fig.suptitle(cells + ' cells: plot ' + str(plot))

            range_use = range(25*plot, 25*(plot + 1))

            for raster, cell_id, a in zip(raster_use[range_use], cell_ids[range_use], ax.reshape(-1)):
                sns.heatmap(raster, ax=a)
                a.axvline(nframes/2, color='g')
                a.set_title('Cell ' + str(cell_id))
            sns.despine(fig=fig)



    def select_events(self, cells, events):
        """Quickly get the appropriate cells and event times to use"""

        # Get appropriate cells to use
        if cells == 'freeze':
            cells_use = self.freeze_cells
        elif cells == 'move':
            cells_use = self.move_cells
        else:
            cells_use = cells

        # Get appropriate events
        if events == 'freeze_onset':
            event_starts = self.freezing_times[:, 0]
        elif events == 'move_onset':
            event_starts = self.freezing_times[:, 1]

        return cells_use, event_starts


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
    Rough - does not consider cells that might predict freezing or motion.
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

    perms = np.random.permutation(len(psa))[0:nperm]  # get nperms

    shuffle_raster = []
    for perm in perms:
        psashuf = np.roll(psa, perm)  # permute psa
        shuffle_raster.append(get_PE_raster(psashuf, buffer_sec=buffer_sec, sr_image=sr_image))

    return np.asarray(shuffle_raster[1:])


def gen_motion_tuning_curve():
    """Function to write to generate neural tuning curves at onset or offset of motion."""
    pass

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
    plot_PSA_w_freezing('Marble29', 'Shock', 1, sort_by='MMI', day2=2
                           , inactive_cells='ignore', plot_corr=True)

    pass