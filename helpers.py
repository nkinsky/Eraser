"""
General helper functions
@author: Nat Kinsky

"""

import numpy as np


def find_epochs(timeseries, thresh=np.finfo(float).eps, omitends=False):
    """
    Get continuous epochs in timeseries that are above a threshold
    :param timeseries: numpy array of values
    :param thresh: value to threshold timeseriew at (default = epsilon)
    :param omitends: include epochs at beginning or end of timeseries
    :return: epochs: nepochs x 2 ndarray with start and end indices of each epoch above thresh
    """

    overthresh = np.greater_equal(timeseries, thresh)  # what is this?

    delta_overthresh = np.diff(np.concatenate(timeseries, np.zeros(1)))
    onsets = np.where(delta_overthresh)
    offsets = np.where(np.bitwise_not(delta_overthresh)) - 1
    nepochs = onsets.shape[0]

    threshepochs = np.zeros((onsets.shape[0], 1))
    if nepochs > 1:
        threshepochs[:, 0] = onsets
        if offsets.shape[1] == nepochs:
            threshepochs[:, 1] = offsets
        elif offsets.shape[1] == nepochs - 1:
            threshepochs[0:-1, 2] = offsets
            threshepochs[:, -1] = timeseries.shape[0]

    if omitends:
        if overthresh(-1):
            nepochs = nepochs -1
            threshepochs = threshepochs[0:-1, :]

        if overthresh(0):
            nepochs = nepochs - 1
            threshepochs = threshepochs[1:, :]

    epochs = threshepochs

    return epochs
