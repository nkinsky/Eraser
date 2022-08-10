import session_directory as sd
import placefield_stability as pfs
import numpy as np
import Placefields as pf
import helpers as hlp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import er_plot_functions as er
import scipy as sp


def get_DI_scores(mouse, arena1, day1, arena2, day2):
    """
    Gets discrimination index correlations between sessions. Note that
    :param mouse:
    :param arena1:
    :param day1:
    :param arena2:
    :param day2:
    """

    # Get mapping between sessions
    neuron_map = pfs.get_neuronmap(mouse, arena1, day1, arena2, day2)
    reg_session = sd.find_eraser_session(mouse, arena2, day2)
    good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)
    good_map = neuron_map[good_map_bool].astype(np.int64)

    # Identify neurons with proper mapping between sessions
    try:
        good_sesh1_ind, _ = np.where(good_map_bool)
    except ValueError:
        good_sesh1_ind = np.where(good_map_bool)
    ngood = len(good_sesh1_ind)

    # load in PF objects for between sessions
    try:
        PF1 = pf.load_pf(mouse, arena1, day1, pf_file='placefields_cm1_manlims_1000shuf.pkl')
        PF2 = pf.load_pf(mouse, arena2, day2, pf_file='placefields_cm1_manlims_1000shuf.pkl')

        # load in event and sampling rates
        fps1 = hlp.get_sampling_rate(PF1)
        ER1, _ = hlp.get_eventrate(PF1.PSAbool_align, fps1)
        fps2 = hlp.get_sampling_rate(PF2)
        ER2, _ = hlp.get_eventrate(PF2.PSAbool_align, fps2)

        # calculate the discrimination index
        DIscores = np.zeros(ngood)
        DIscores = (ER1[good_sesh1_ind] - ER2[good_map]) / (ER1[good_sesh1_ind] + ER2[good_map])

        return DIscores

    except FileNotFoundError:  # If pf files are missing, return NaN
        return np.nan


def get_on_off_cells(mouse, arena1, day1, arena2, day2):
     # Get mapping between sessions and use Nat's classify cells function to return good_map_bool, silent_ind, and new_ind
     neuron_map = pfs.get_neuronmap(mouse, arena1, day1, arena2, day2)
     reg_session = sd.find_eraser_session(mouse, arena2, day2)
     good_map_bool, silent_ind, new_ind = pfs.classify_cells(neuron_map, reg_session)

     both_cells = int(sum(good_map_bool == True))  # good_map_bool == true sum will yield total number of neurons that stayed on in both sessions
     on_cells = len(new_ind)  # len of new_ind corresponds to the number of new neurons appearing in session 2
     off_cells = len(silent_ind)  # len of silent_ind corresponds to the number of new neurons appearing in session 2

     return both_cells, on_cells, off_cells


def DI_CC_scatter(mice):
    nmice = len(mice)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    DIarray_total = []
    on_off_norm_total = np.empty((nmice, 3))
    idc = 0
    idr = 0
    days = [1, 2, 7]
    for mouse in mice:
        for day in days:
            DIarray =[]
            # this should spit out the mouse with issues, then you can zero in after that via debugging
            # (make sure to comment out try/except statement first)
            #try:
            DIval = er.DIFreeze(mouse, [day])
            DIarray += DIval
            DIarray_total += DIval
            cc_both, cc_on, cc_off = get_on_off_cells(mouse, "Open", day, "Shock", day)
            on_off_norm = np.divide((cc_on + cc_off), (cc_on + cc_off + cc_both))
            on_off_norm_total[idr, idc] = on_off_norm
            ax.scatter(DIarray, on_off_norm)
            #except ValueError:
                #print('Error in ' + mouse)
            idc += 1
        idc = 0
        idr += 1
    title = input("What is your title?")
    plt.title(title)
    plt.xlabel("DI_Freeze")
    plt.ylabel("Normalized cell count for on and off cells")
    try:
        on_off_norm_total = on_off_norm_total.reshape(-1)
        DIarray_total = np.array(DIarray_total)
        good_pts_bool = np.bitwise_and(~np.isnan(DIarray_total), ~np.isnan(on_off_norm_total.reshape(-1)))
        z = np.polyfit(DIarray_total[good_pts_bool], on_off_norm_total[good_pts_bool], 1)
        p = np.poly1d(z)
        plt.plot(ax.get_xlim(), p(ax.get_xlim()), "r--")
    except ValueError:
          print('Error in trend line')
    Corrcoef = sp.stats.pearsonr(DIarray_total[good_pts_bool], on_off_norm_total[good_pts_bool])
    plt.annotate("Pearson Correlation Coefficients: R = " + str(round(Corrcoef[0],3)) + "; P = " + str(round(Corrcoef[1],3)), xy=(0.15, 0.95), xycoords='axes fraction')
    plt.show()
    return Corrcoef


if __name__ == '__main__':
    DIneurons = get_DI_scores('Marble06', 'Open', 1, 'Shock', 1)

pass