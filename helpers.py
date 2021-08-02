"""
General helper functions
@author: Nat Kinsky

"""

import numpy as np
import matplotlib.pyplot as plt


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


def get_eventrate(PSAbool_align,fps):
    """
    gets event rate and event probability for calcium rasters
    :param timeseries: boolean
    :return:
    """
    event_rate = []
    event_prob = []
    for x in list(range(len(PSAbool_align))):
        epochs = find_epochs(PSAbool_align[x,:])
        active_frames = 0
        total_frames = len(PSAbool_align[x,:])
        transients = len(epochs)
        for x in range(len(epochs)):
            active_frames += epochs[x,1] - epochs[x,0]
        event_rate += [transients/(total_frames/(60*fps))] # transients per minute
        event_prob += [(active_frames/total_frames)*100]
    event_rate = np.array(event_rate)
    event_prob = np.array(event_prob)
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


if __name__ == '__main__':
    import Placefields as pf
    mouse, arena, day = 'Marble07', 'Shock', -2
    PF = pf.load_pf(mouse, arena, day)
    sortPSA(PF.PSAbool_align)
    pass