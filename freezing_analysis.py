import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers


def get_freezing_times(mouse, arena, day, **kwargs):
    """Identify chunks of frames and timestamps during which the mouse was freezing

    :param mouse: str
    :param arena: 'Open' or 'Shock'
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param kwargs: see er_plot_functions.detect_freezing()
    :return: freezing_epochs: list of start and end indices of each freezing epoch in behavioral video
             freezing_times: list of start and end times of each freezing epoch
    """
    dir_use = erp.get_dir(mouse, arena, day)

    video_t = erp.get_timestamps(str(dir_use))
    freezing, velocity = erp.detect_freezing(str(dir_use), arena=arena, **kwargs)
    video_t = video_t[:-1]  # Chop off last timepoint to make this the same length as freezing and velocity arrays

    # convert freezing indices to timestamps
    freezing_epochs = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_epochs]

    return freezing_epochs, freezing_times


def align_freezing_to_PSA(PSAbool, sr_image, freezing, video_t):
    """
    Align freezing times to neural data.
    :param PSAbool: nneurons x nframes_imaging boolean ndarray of putative spiking activity
    :param sr_image: frames/sec
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
        freeze_bool[np.bitwise_and(PSAtime >= freeze_time[0], PSAtime < freeze_time[1])] = True

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


def plot_PSA_w_freezing(mouse, arena, day, sort_by='first_event', day2=False, ax=None, **kwargs):
    """Plot *raw* calcium event rasters across whole session with velocity trace overlaid in red and freezing epochs
    overlaid in green. Can sort by various interesting metrics and apply that across days.
    :param mouse: str
    :param arena: 'Open' or 'Shock'
    :param day: int from [-2, -1, 0, 4, 1, 2, 7]
    :param sort_by: how to sort neurons, options are: 'first_event', 'move_event_rate', 'freeze_event_rate', 'MMI', or None
    :param day2: 2nd day to use sorted by 1st day sorting (new neurons at the end). False (default) = plot one day only.
    :param ax: axes to plot into, default (None) = create new figure and axes.
    :param kwargs: freezing related parameters for calculating freezing with all 'sort_by' options except 'first_event'.
    See er_plot_functions.detect_freezing(). Can also toggle 'batch_map_use' to True or False for sorting across days.
    :return: ax: axes or list of plotting axes
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
    def plotPSAoverlay(PSAuse, sr_image, video_t, velocity, freezing_times, mouse, arena, day, ax):
        SFvel = 4  # Factor to scale velocity by for overlay below

        sns.heatmap(data=PSAuse, ax=ax, xticklabels=1000, yticklabels=50)
        nneurons = PSAuse.shape[0]
        ax.plot(video_t * sr_image, velocity * -SFvel + nneurons / 2, color=[1, 0, 0, 0.5], linewidth=1)
        # ax.plot(video_t[freezing] * PF.sr_image, velocity[freezing] * -SFvel + nneurons / 2, 'b*', alpha=0.5)

        for freeze_time in freezing_times:
            ax.axvspan(freeze_time[0] * sr_image, freeze_time[1] * sr_image, color=[0, 1, 0, 0.4])

        # Pretty things up and label
        ax.tick_params(axis='y', rotation=0)
        ax.set_xticklabels([int(int(label.get_text()) / sr_image) for label in ax.get_xticklabels()])
        ax.set_title(mouse + ' ' + arena + ' Day ' + str(day) + ': ' + str(sort_by) + ' sort')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')

    # Sorting sub-function
    def sort_PSA(mouse, arena, day, PSAbool_in, freeze_bool, sort_by, **kwargs):
        """Sort PSA according in `sort_by` metric"""

        # Next, sort PSAbool appropriately
        nneurons = PSAbool_in.shape[0]
        if sort_by is None:
            sort_array = np.arange(0, nneurons)
        elif sort_by == 'first_event':  # Sort by first calcium event time
            PSAuse, sort_array = helpers.sortPSA(PSAbool_in)
        elif sort_by == 'move_event_rate':
            sort_array = move_event_rate(PSAbool_in, freeze_bool)
        elif sort_by == 'freeze_event_rate':
            sort_array = move_event_rate(PSAbool_in, freeze_bool)
        elif sort_by == 'MMI':
            sort_array = motion_modulation_index(mouse, arena, day, **kwargs)

        if sort_by != 'first_event':
            PSAuse = helpers.sortPSA(PSAbool_align, sort_by=sort_array)

        return PSAuse, sort_array

    # First get PSA and velocity data
    PSAbool_align, sr_image, video_t, velocity, freezing_times, freeze_bool = \
        getPSA_and_freezing(mouse, arena, day, **kwargs)

    # Now sort
    PSAuse, sort_array = sort_PSA(mouse, arena, day, PSAbool_align, freeze_bool, sort_by, **kwargs)

    # Now set up plotting
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        nplots = 2 if day2 else 1
        gs = gridspec.GridSpec(nrows=1, ncols=nplots, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])

    # append a block of NaN rows with 0th dimension = # new neurons if plotting across days
    if day2:
        # register and apply sorting to sort_array from above.
        neuron_map = pfs.get_neuronmap(mouse, arena, day, arena, day2, **kwargs)
        reg_session = sd.find_eraser_session(mouse, arena, day2)
        good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)

        nframes = PSAuse.shape[1]

        PSAuse = np.concatenate((PSAuse, np.ones((len(new_ind), nframes))*np.nan))

    # Finally plot
    plotPSAoverlay(PSAuse, sr_image, video_t, velocity, freezing_times, mouse, arena, day, ax=ax1)

    # Now plot 2nd day if looking across days!
    if day2:
        PSAbool_align2, sr_image2, video_t2, velocity2, freezing_times2, freeze_bool2 = \
            getPSA_and_freezing(mouse, arena, day2, **kwargs)

        # next register and apply sorting to sort_array from above.
        neuron_map = pfs.get_neuronmap(mouse, arena, day, arena, day2, **kwargs)
        reg_session = sd.find_eraser_session(mouse, arena, day2)
        good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)

        # now sort 2nd session cells by original session order
        sort_array_reg = neuron_map[sort_array]

        # Now sort new cells by same metric and append cell ids to sort_array_reg
        PSAuse2_reg, sort_array2 = sort_PSA(mouse, arena, day2, PSAbool_align2, freeze_bool2, sort_by, **kwargs)

        # NRK todo: bug here - need to create 2nd psabool array of new neurons and concatenate to PSAbooluse2 above rather than creating an index to sort by OR we need to use the PSA_sortby function above
        PSAreg = []
        nframes_reg = PSAuse2_reg.shape[1]
        for ind in sort_array_reg:
            if not np.isnan(ind):  # Add actual cell activity
                psa_to_add = PSAbool_align2[int(ind)]
            else:  # Add in all nans if not active the next day
                psa_to_add = np.ones(nframes_reg)*np.nan
            PSAreg.append(psa_to_add)  # append psa
        PSAreg = np.asarray(PSAreg)  # convert from list to array

        # Now add in new cells at bottom
        PSAreg = np.concatenate((PSAreg, PSAuse2_reg[new_ind]), axis=0)

        # Finally plot side-by-side!
        ax2 = fig.add_subplot(gs[0, 1])
        plotPSAoverlay(PSAreg, sr_image2, video_t2, velocity2, freezing_times2, mouse, arena, day2, ax=ax2)

    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    plot_PSA_w_freezing('Marble07', 'Shock', -2, sort_by='first_event', day2=-1)

    pass