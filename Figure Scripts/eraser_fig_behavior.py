# Behavioral plots here!

import eraser_reference as err
import er_plot_functions as er
import scipy.stats as s
import numpy as np

## Get difference in context-specific freezing between groups at LTM memory time points!
_, _, fratio_cont = er.plot_all_freezing(err.control_mice)
_, _, fratio_ani = er.plot_all_freezing(err.ani_mice)

## Run stats
pval_bwgroup_1sidedt_specific = np.ones(6)*np.nan
pval_bwgroup_1sidedt_specific_no29 = np.ones(6)*np.nan
pval_bwgroup_1sidedt_shockonly = np.ones(6)*np.nan # bw group direct, no accounting for specificity
for id in range(0, 6):
    stats, pval = s.ttest_ind(fratio_ani[1, id, :].reshape(-1) - fratio_ani[0, id, :].reshape(-1),
                              fratio_cont[1, id, :].reshape(-1) - fratio_cont[0, id, :].reshape(-1),
                              nan_policy='omit')
    # Marble29 seems to freeze a LOT in the shock arena prior to being shocked. Check if real later, exclude for now.
    stats_no29, pval_no29 = s.ttest_ind(fratio_ani[1, id, :].reshape(-1) - fratio_ani[0, id, :].reshape(-1),
                              fratio_cont[1, id, 0:7].reshape(-1) - fratio_cont[0, id, 0:7].reshape(-1),
                              nan_policy='omit')

    statss, pvals = s.ttest_ind(fratio_ani[1, id, :], fratio_cont[1, id, :], nan_policy='omit')
    pval_bwgroup_1sidedt_shockonly[id] = pvals/2

    # Hypothesis is that ani freezing is less than control guys relative to neutral arena
    # So if ani guys mean freezing relative to shock is < 0 (i.e. if stats < 0), divide the pvalue by 2
    # If it's actually larger, then test fails and p = 1-pval/2
    if stats < 0:
        pval_bwgroup_1sidedt_specific[id] = pval/2
    elif stats > 0:
        pval_bwgroup_1sidedt_specific[id] = 1 - pval/2

    if stats_no29 < 0:
        pval_bwgroup_1sidedt_specific_no29[id] = pval_no29 / 2
    elif stats_no29 > 0:
        pval_bwgroup_1sidedt_specific_no29[id] = 1 - pval_no29 / 2
# Good for one-sided test!



## Get differences between day -1 and day 1 for shock and anisomycin groups



##  Now between day -1 and day 1 for each group, shock arena only

statsc, pvalc = s.ttest_rel(fratio_cont[1, 1, :], fratio_cont[1, 3, :])
pvalc = pvalc/2

# try to exclude Marble29 who seems to freeze a bunch even BEFORE being shocked...
statsc_no29, pvalc_no29 = s.ttest_rel(fratio_cont[1, 1, 0:7], fratio_cont[1, 3, 0:7])
pvalc_no29 = pvalc_no29/2

statsa, pvala = s.ttest_rel(fratio_ani[1, 1, :], fratio_ani[1, 3, :])
pvala = pvala/2


## Plot the above - day -1 to day 1 in shock arena, then day shock-freezing compared between
# each arena


