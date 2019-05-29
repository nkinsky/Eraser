"""
General helper functions
@author: Nat Kinsky

"""

import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Placefields import load_pf
    PF = load_pf("Marble24", "Shock", "-1", pf_file='placefields_cm1_manlims.pkl')
    ans = get_sampling_rate(PF)
    x , y = get_eventrate(PF.PSAbool_align, ans)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    ax.scatter(x, y)
    plt.show()
    # test comment by evan
    pass

# import session_directory as sd
# import os
# import pickle
# dir_use = sd.find_eraser_directory('Marble24','Shock','-1')
# os.path.join(dir_use,'placefields_cml_manlims.pkl')
# pf_file = os.path.join(dir_use, 'placefields_cm1_manlims.pkl')
# with open(pf_file, 'rb') as file:
#     PF = pickle.load(file)