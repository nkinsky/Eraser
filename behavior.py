# Parse behavior in contextual fear conditioning task
# Created by Nat Kinsky 2020 March 04, 3:00pm CST

import numpy as np
import Placefields as pf  ## Eraser module

def get_thigmotaxis(mouse, day, arena, pf_file='placefields_cm1_manlims_1000shuf.pkl'):
    """
    Get thigmotaxis for a given session.  Wrapper function for calc_thigmotaxis specific to eraser and
    assumes pre-cultivated position data.
    :param mouse: eraser specific session parameter
    :param day: eraser specific session parameter:
    :param arena: eraser specific session parameter
    :param pf_file (optional): placefield file to load pre-aligned position data from.
    default = pf_file='placefields_cm1_manlims_1000shuf.pkl'
    :return: thigmo_ratio: Defined as ratio of time spent in outer part of the arena versus total time in
    arena in line with Wang et al. (J Neuro, 2012).
    """

    pfo = pf.load_pf(mouse, arena, day, pf_file=pf_file)
    thigmo_ratio = calc_thigmotaxis(pfo.occmap)

    return thigmo_ratio


def calc_thigmotaxis(occmap, nbins_pad = 1):
    """
    Calculate thigmotaxis, defined as time spend in the outer part of the arena over total time in the arena. General
    function. Requires cultivating data to calculate an occupancy map. Based on Wang et al. (J Neuro, 2012) which used a
    circular arena and divided it into three equally space rings.
    :param occmap: 2-d occupancy map with # frames or time spent in each bin across the whole session.
    :param nbins_pad (optional): # bins in 2-d occupancy map at the edges to exclude when calculating arena extents.
    default = 1.
    :return: thigmo_ratio: Defined as ratio of time spent in outer 3rd of the arena versus total time in
    arena.
    """

    # Get extent of mouse occupancy
    nbinsx, nbinsy = occmap.shape
    xspan = nbinsx - 2*nbins_pad
    yspan = nbinsy - 2*nbins_pad

    # Calculate extent of outer ring and make boolean
    outer_ring = np.zeros(occmap.shape)
    outer_x = np.floor(xspan/5)
    outer_y = np.floor(yspan/5)
    outer_ring[nbins_pad:nbins_pad + outer_x, nbins_pad:-nbins_pad] = 1  # top part of ring
    outer_ring[-(nbins_pad + outer_x):-nbins_pad, nbins_pad:-nbins_pad] = 1  # bottom part of ring
    outer_ring[nbins_pad:-nbins_pad, nbins_pad:nbins_pad+outer_y] = 1  # left part of ring
    outer_ring[nbins_pad:-nbins_pad, -(nbins_pad + outer_y):-nbins_pad] = 1  # right part of ring

    # Calculate thigmotaxis
    occ_outer = occmap[outer_ring].sum()
    occ_total = occmap.sum()
    thigmo_ratio = occ_outer/occ_total

    return thigmo_ratio

# def PCA_fear(freezing, thigmo_ratio, speed):  # Should these variables be a single time point or multiple time points?
#
