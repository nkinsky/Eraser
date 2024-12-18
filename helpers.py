"""
General helper functions
@author: Nat Kinsky

"""

import numpy as np
import matplotlib.pyplot as plt
from session_directory import find_eraser_directory as get_dir
from scipy.io import loadmat
from pathlib import Path
from collections.abc import Iterable


def range_to_slice(range_var):
    """Convert range to slice"""
    assert isinstance(range_var, range)

    return slice(range_var.start, range_var.stop, range_var.step)


def set_ticks_to_lim(ax: plt.axes, x: bool = True, y: bool = True):
    """
    Sets ticks and labels to max/min of values on a plot
    :param ax: axes to clean up
    :param x, y: bool, True = set to lims, False = leave alone
    :return:
    """

    if x:
        ax.set_xticks(ax.get_xticks()[[0, -1]])

    if y:
        ax.set_yticks(ax.get_yticks()[[0, -1]])

    return ax

def get_ROIs(mouse: str, arena: str, day: int in [-2, -1, 0, 1, 4, 2, 7], **kwargs):
    """
    Get neuron ROI array.
    :param mouse: str of the format 'Marble##'
    :param arena: str in ['Open', Shock']
    :param day: int in [-2, -1, 0, 4, 1, 2, 7]
    :param **kwargs: input to session_directory.find_eraser_directory
    :return: rois an ncells x npixx x npixy ndarray of neuron ROIs
    """
    dir_use = Path(get_dir(mouse, arena, day, **kwargs))
    neural_data = loadmat(dir_use / 'FinalOutput.mat')
    neuron_image = neural_data['NeuronImage']

    # Convert from a series of xpix x ypix arrays to a 3d ndarray
    assert neuron_image.shape[0] == 1, 'Inappropriate format for NeuronImage in FinalOutput.mat - write code!'
    rois = np.asarray([roi for roi in neuron_image[0]])

    return rois


def get_CI(data, pct=95):
    """Get mean and confidence intervals (at pct specified by input) for input data (size= n,).
    :returns: ndarray of [CIbottom, mean, CItop]"""
    qtop = 1 - (100 - pct) / 2 / 100
    qbot = (100 - pct) / 2 / 100
    return np.quantile(data, [qbot, 0.5, qtop])


def mean_CI(data_list, pct=95):
    """Get means of mean and upper/lower confidence interval for all data in data_list
    e.g. if len(data_list)=4, spits out the mean of those 4 data arrays.
    """
    CIall = [get_CI(data, pct=pct) for data in data_list]

    return np.nanmean(np.vstack(CIall), 0)


def match_ax_lims(axes, type='y'):
    """Match axis limits across a list of different figure axes"""
    # Set up everything
    get_funcs, set_funcs = ['get_xlim', 'get_ylim'], ['set_xlim', 'set_ylim']
    assert type in ['x', 'y']
    getfuncstr = get_funcs[np.where([type == _ for _ in ['x', 'y']])[0][0]]
    setfuncstr = set_funcs[np.where([type == _ for _ in ['x', 'y']])[0][0]]

    # Now aggregate all min/max limits
    mins, maxs = [], []
    for ax in axes:
        func_use = getattr(ax, getfuncstr)  # pull out appropriate axis limit function

        mins.append(func_use()[0])
        maxs.append(func_use()[1])

    alims = [np.min(mins), np.max(maxs)]

    # Now set limits for all axes
    for ax in axes:
        func_use = getattr(ax, setfuncstr)  # pull out appropriate axis limit function
        func_use(alims)

    return alims

def find_epochs(timeseries, thresh=np.finfo(float).eps, omitends=False):
    """
    Get continuous epochs in timeseries that are above a threshold
    :param timeseries: numpy array of values
    :param thresh: value to threshold timeseriew at (default = epsilon)
    :param omitends: include epochs at beginning or end of timeseries
    :return: epochs: nepochs x 2 ndarray with start and end indices of each epoch above thresh
    """

    overthresh = np.greater_equal(timeseries, thresh)  # takes a timeseries containing zeros and ones that indicate when a cell is firing and returns a boolean value corresponding to whether that cell was active or not?

    delta_overthresh = np.diff(np.concatenate((np.zeros(1), overthresh))) # creates an array that uses a 1 to delineate when cells come online; a -1 for when cells go offline; and a 0 when cells are not changing their state
    onsets = np.where(delta_overthresh == 1)[0] # creates an array containing the time points during which a cell is coming online
    offsets = np.where(delta_overthresh == - 1)[0] - 1 # creates an array containing the time points during which a cell goes offline
    nepochs = onsets.shape[0] # calculates the number of onsets there are and returns an integer that represents the total number of calcium events that occured

    threshepochs = np.zeros((onsets.shape[0], 2)) # creates an array with two columns and as many rows as there are epochs
    # makes sure onsets == offsets i.e if you end in an offset make sure that you have the same numbers of corresponding onsets
    if nepochs > 1:
        threshepochs[:, 0] = onsets
        if offsets.shape[0] == nepochs:
            threshepochs[:, 1] = offsets
        elif offsets.shape[0] == nepochs - 1: #if there is an overthreshold at the end of the video add an offset corresponding to the end of the timeseries
            threshepochs[0:-1, 1] = offsets
            threshepochs[-1, 1] = timeseries.shape[0]

    if omitends:
        if overthresh[-1]: #if overthresh at the end of the video
            nepochs = nepochs - 1 #delete the last epoch
            threshepochs = threshepochs[0:-1, :]

        if overthresh[0]: #if overthresh at the beginning of the video
            nepochs = nepochs - 1 #delete the first epoch
            threshepochs = threshepochs[1:, :]

    epochs = threshepochs

    return epochs


def set_all_lims(ax, xlims, ylims):
    """
    Sets all axes to the same limits
    :param ax: matplot.pyplot axes
    :param xlims: 1 x 2 array.
    :param ylims: 1 x 2 array
    :return:
    """
    for a in ax.reshape(-1):
        a.set_xlim(xlims)
        a.set_ylim(ylims)


def set_all_lim_range(ax, range, xmin, ymin):
    """
    Sets all axes to have the same range of values in x and y directions but with different start and ends points.
    :param ax: matplot.pyplot axes
    :param range: 1 x 2 array with x, y range
    :param xmin/ymin: #axes array with x and y min values
    :return:
    """
    for ida, a in enumerate(ax.reshape(-1)):
        a.set_xlim([xmin[ida], range[0] + xmin[ida]])
        a.set_ylim([ymin[ida], range[1] + ymin[ida]])


def get_sampling_rate(PF):
    try:
        sr_image = PF.sr_image
    except(AttributeError):
        diff_1 = len(PF.PSAbool_align[0]) - 6000
        diff_2 = len(PF.PSAbool_align[0]) - 12000
        if abs(diff_1) > abs(diff_2):
            sr_image = 20
        if abs(diff_2) > abs(diff_1):
            sr_image = 10
    # If sr_image is not found in PF print out a warning to the screen
        print("sample rate could not be located in PF")
    return sr_image


def get_eventrate(PSAbool_align, fps):
    """
    gets event rate and event probability for calcium rasters
    :param timeseries: boolean
    :return:
    """
    dur_min = PSAbool_align.shape[1] / (60 * fps)
    event_rate = []
    for psa in PSAbool_align:
        event_rate.append(len(find_epochs(psa)) / dur_min)
    event_prob = PSAbool_align.sum(axis=1) / PSAbool_align.shape[1]
    event_rate = np.array(event_rate)
    return event_rate, event_prob


def plot_prob_hist(array):
    weights = np.ones_like(array) / float(len(array))
    n, bins, patches = plt.hist(abs(array), bins=10, range=(0, 1), weights=weights)
    plt.show()
    return n, bins, patches
# def plot_ERvsEP(ER,EP):
#
#     PF = load_pf(str(mouse), "Shock", str(x), pf_file='placefields_cm1_manlims.pkl')
#     data += [gen_ERvsRP(PF.PSAbool_align[0,:])]
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
#     x, y = data
#     ax.scatter(x, y)
#     plt.show()


def sortPSA(PSAbool, sort_by=None):
    """
    Sorts putative-spiking activity boolean array (PSAbool) by time of first calcium event. Or by whatever you want
    if you use 'sort_by' parameter.
    :param PSAbool: nneurons x nframes boolean ndarray of putative spiking activity
    :param sort_by: if not None, alternative metric to sort cells by - default = None.
    :return: PSAsort: sorted PSAbool
    """
    nneurons,_ = PSAbool.shape

    if sort_by is None:
        # First identify active and inactive (perhaps not active after speed thresholding) neurons
        inactive_neurons = np.where(np.invert(np.any(PSAbool, 1)))
        active_neurons = np.where(np.any(PSAbool, 1))[0]

        n_events = np.nonzero(PSAbool)  # get indices of all calcium events (1 = neuron#, 2 = frame#)

        # Get time of first calcium event and sort
        onset_frame = np.asarray([np.min(n_events[1][n_events[0] == neuron])
                                  for neuron in active_neurons])
        sort_ind_active = np.argsort(onset_frame)
        sort_ind = np.append(active_neurons[sort_ind_active], inactive_neurons)
    else:
        sort_ind = np.argsort(sort_by)

    PSAsort = PSAbool[sort_ind, :]

    return PSAsort, sort_ind


def sortPSAbyarray(PSAin, sort_ind):
    """sorts PSAbool by the array indices in sort_ind. Sets any NaNs to all NaN rows in PSAsort_out"""

    # Step through each neuron in sort_ind and grab the appopriate row from PSAin.
    nneurons, nframes = PSAin.shape
    PSAsort = []
    for neuron in sort_ind:

        # Get neural activity, making all frames NaN if sort_ind is Nan
        if np.isnan(neuron):
            psa = np.ones(nframes)*np.nan
        else:
            psa = PSAin[neuron]

        PSAsort.append(psa)

    return PSAsort


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index. Taken directly from stackoverflow:
    https://stackoverflow.com/questions/4494404/find-large-number-of-
    consecutive-values-fulfilling-condition-in-a-numpy-array"""

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def get_transient_peaks(rawtrace, psabool):
    """Gets and returns the peak height of each transient"""
    idt = contiguous_regions(psabool)  # Parse out transient times
    peak_heights = []
    for start, end in idt:
        peak_heights.append(np.max(rawtrace[start:end]))

    return np.array(peak_heights)


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    if len(all_) == 1:
        return np.array(all_, dtype=int).squeeze()
    else:
        return np.around(np.mean(all_)).astype(int).squeeze()

if __name__ == '__main__':
    import Placefields as pf
    mouse, arena, day = 'Marble07', 'Shock', -2
    PF = pf.load_pf(mouse, arena, day)
    sortPSA(PF.PSAbool_align)
    pass