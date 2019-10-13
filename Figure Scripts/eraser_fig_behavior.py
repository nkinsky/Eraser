## Behavioral plots here!

import eraser_reference as err
import er_plot_functions as er
import scipy.stats as s
import numpy as np
import matplotlib.pyplot as plt
from os import path

# Make text save as whole words
plt.rcParams['pdf.fonttype'] = 42

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

## Get difference in context-specific freezing between groups at LTM memory time points!
figc, axc, fratio_cont = er.plot_all_freezing(err.control_mice)
axc.set_ylim([0, 0.80])
axc.set_title('Control')
figc.savefig(path.join(plot_dir, 'Control Group All Freezing Plot.pdf'))

# Exclude 29 who is freezing a bunch before even being shocked in shock arena
figcn29, axcn29, fratio_contn29 = er.plot_all_freezing(err.control_mice[0:-1])
axcn29.set_ylim([0, 0.80])
axcn29.set_title('Control')
figcn29.savefig(path.join(plot_dir, 'Control Group All Freezing Plot excluding Marble29.pdf'))

figa, axa, fratio_ani = er.plot_all_freezing(err.ani_mice)
axa.set_ylim([0, 0.80])
axa.set_title('Anisomycin')
figa.savefig(path.join(plot_dir, 'Anisomycin Group All Freezing Plot.pdf'))
_, axg, fratio_gen = er.plot_all_freezing(err.generalizers)
axg.set_title('*Generalizers*')
_, axd, fratio_disc = er.plot_all_freezing(err.discriminators)
axd.set_title('Discriminators')

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
from plot_functions import scatter_box, scatter_bar

# Scatterbox form
figc, axc = scatter_box([100*fratio_cont[1, 1, 0:6], 100*fratio_cont[1, 3, 0:6],
                         100*fratio_cont[0, 3, 0:6]],
                        xlabels=['Day -1', 'Day 1', 'Day 1 Neutral'], ylabel='Freezing (%)',
                        alpha=0.5)
axc.set_title('Control Mice')

figa, axa = scatter_box([100*fratio_ani[1, 1, [0, 2, 3, 5]],
                         100*fratio_ani[1, 3, [0, 2, 3, 5]],
                         100 * fratio_ani[0, 3:6, [0, 3, 5]][
                             np.logical_not(np.isnan(100 * fratio_ani[0, 3:6, [0, 3, 5]]))]],
                        xlabels=['Day -1', 'Day 1', 'Day 1 Neutral'], ylabel='Freezing (%)',
                        alpha=0.5)
axa.set_title('Anisomycin Mice')

# scatter_bar form
figc, axc = scatter_bar([100*fratio_cont[1, 1, 0:6], 100*fratio_cont[1, 3, 0:6],
                         100*fratio_cont[0, 3, 0:6]],
                        xlabels=['Day -1', 'Day 1', 'Days 1-7 Neutral'],
                        ylabel='Freezing (%)', alpha=0.5)
axc.set_title('Control Mice')

figab, axab = scatter_bar([100*fratio_ani[1, 1, [0, 2, 3, 5]],
                         100*fratio_ani[1, 3, [0, 2, 3, 5]],
                         100 * fratio_ani[0, 3:6, [0, 3, 5]][
                             np.logical_not(np.isnan(100 * fratio_ani[0, 3:6, [0, 3, 5]]))]],
                         xlabels=['Day -1', 'Day 1', 'Days 1-7 Neutral'],
                          ylabel='Freezing (%)', alpha=0.5)
axab.set_title('Anisomycin Mice')


##  Now between day -1 and day 1 for each group, shock arena only

validc_bool = np.bitwise_and(~np.isnan(fratio_cont[1, 1, :]),
                             ~np.isnan(fratio_cont[1, 3, :]))
statsc_paired, pvalc_paired = s.ttest_rel(fratio_cont[1, 1, validc_bool],
                                          fratio_cont[1, 3, validc_bool])
statsc_ind, pvalc_ind = s.ttest_ind(fratio_cont[1, 1, ~np.isnan(fratio_cont[1, 1, :])],
                                    fratio_cont[1, 3, ~np.isnan(fratio_cont[1, 3, :])])


# try to exclude Marble29 who seems to freeze a bunch even BEFORE being shocked...
validc_no29_bool = validc_bool.copy()
validc_no29_bool[8] = False
statsc_no29, pvalc_no29 = s.ttest_rel(fratio_cont[1, 1, validc_no29_bool],
                                      fratio_cont[1, 3, validc_no29_bool])
# pvalc_no29 = pvalc_no29/2

valida_bool = np.bitwise_and(~np.isnan(fratio_ani[1, 1, :]),
                             ~np.isnan(fratio_ani[1, 3, :]))
statsa, pvala = s.ttest_rel(fratio_ani[1, 1, valida_bool], fratio_ani[1, 3, valida_bool])
# pvala = pvala/2


## Plot the above - day -1 to day 1 in shock arena, then day shock-freezing compared between
# each arena


