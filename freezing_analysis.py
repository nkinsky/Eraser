import numpy as np
import er_plot_functions as erp


def align_freezing_to_PSA(PSAbool, sr_image, freezing, video_t):

    # First get imaging parameters and freezing in behavioral video timestamps
    nneurons, nframes = PSAbool.shape
    freezing_epochs = erp.get_freezing_epochs(freezing)
    freezing_times = [[video_t[epoch[0]], video_t[epoch[1]]] for epoch in freezing_epochs]

    # Set up boolean to match neural data shape
    freezingPSA = np.zeros(nframes, dtype='bool')
    PSAtime = np.arange(0, nframes)/sr_image

    # Interpolate freezing times in video time to imaging time
    for freeze_time in freezing_times:
        freezingPSA[np.bitwise_and(PSAtime >= freeze_time[0], PSAtime < freeze_time[1])] = True

    return freezingPSA