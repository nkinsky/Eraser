# Behavioral plots here!

import eraser_reference as err
import er_plot_functions as er
import scipy.stats as s
import numpy as np

## Get difference in context-specific freezing between groups at LTM memory time points!
_, _, fratio_cont = er.plot_all_freezing(err.control_mice_good)
_, _, fratio_ani = er.plot_all_freezing(err.ani_mice_good)

## Run stats
pval_bwgroup_1sidedt_specific = np.ones(6)*np.nan
for id in range(0, 6):
    stats, pval = s.ttest_ind(fratio_ani[1, id, :].reshape(-1) - fratio_ani[0, id, :].reshape(-1),
                              fratio_cont[1, id, :].reshape(-1) - fratio_cont[0, id, :].reshape(-1),
                              nan_policy='omit')

    # Hypothesis is that ani freezing is less than control guys relative to neutral arena
    # So if ani guys mean freezing relative to shock is < 0 (i.e. if stats < 0), divide the pvalue by 2
    # If it's actually larger, then test fails and p = 1-pval/2
    if stats < 0:
        pval_bwgroup_1sidedt_specific[id] = pval/2
    elif stats > 0:
        pval_bwgroup_1sidedt_specific[id] = 1 - pval/2
# Good for one-sided test!



