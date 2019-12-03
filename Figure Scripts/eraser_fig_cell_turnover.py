## Get overlap ratios for each session to the first
import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import numpy as np
import eraser_reference as err

# Make text save as whole words
plt.rcParams['pdf.fonttype'] = 42

group_title = 'Control'
if group_title is 'Control':
    mice = err.control_mice_good
elif group_title is 'Anisomycin':
    mice = err.ani_mice_good

days = [-1, 0, 4, 1, 2]  # use this to look at change from day 0 to 4hr sesh (good in control arena only)
# days = [-1, 4, 1, 2, 7]  # legit way - use to look at change across all days with 10 min sessions
arenas = ['Shock', 'Open']
oratio1 = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratio2 = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratioboth = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratiomin = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratiomax = np.ones((len(mice), len(days), len(arenas)))*np.nan

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
plot_ind = True
pathname = err.pathname  # Get plotting folder
for idm, mouse in enumerate(mice):
    for idd, day in enumerate(days):
        for ida, arena in enumerate(arenas):
            try:
                oratio1[idm, idd, ida], oratio2[idm, idd, ida], oratioboth[idm, idd, ida], \
                    oratiomin[idm, idd, ida], oratiomax[idm, idd, ida] = \
                    pfs.get_overlap(mouse, 'Shock', -2, arena, day)
            except TypeError:
                print('Missing reg file for ' + mouse + ' Day ' + str(day) + ' ' + arena)

    if plot_ind:
        fig, ax = er.plot_overlaps(oratio1[idm, :, :], days=days)
        fig.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + '-' + mouse + '.pdf'))
        plt.close(fig)

fig, ax = er.plot_overlaps(oratioboth, days=days)
ax.set_title(group_title + ' Both normalized')
fig.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Both normalized.pdf'))
fig2, ax2 = er.plot_overlaps(oratiomax, days=days)
ax2.set_title(group_title + ' Max normalized')
fig2.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Max normalized.pdf'))
fig3, ax3 = er.plot_overlaps(oratiomin, days=days)
ax3.set_title(group_title + ' Min normalized')
fig3.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Min normalized.pdf'))

# Now combine!
oratioboth_comb = np.expand_dims(np.concatenate((oratioboth[:, :, 0], oratioboth[:, :, 1]), 0), 2)
figc, axc = er.plot_overlaps(oratioboth_comb, days=days)
axc.set_title(group_title + ' Combined Both normalized')
figc.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Combined Min normalized.pdf'))

## Plot scatterboxes for change from day -1 to 4hr and 4hr to day 1
from plot_functions import scatter_box
import scipy.stats as sstats

# Remove marble14/20 for now, fix once data points are recovered
if group_title is 'Control':
    oratio_good = oratioboth[[0, 1, 2, 3, 5], :, :]
    oratio_goodc = oratio_good
elif group_title is 'Anisomycin':
    oratio_good = oratioboth[[0, 1, 3], :, :]
    oratio_gooda = oratio_good

fig, ax = scatter_box([oratio_good[:, 0, :].reshape(-1) - oratio_good[:, 1, :].reshape(-1),
                       oratio_good[:, 1, :].reshape(-1) - oratio_good[:, 2, :].reshape(-1)],
                      xlabels=['Day -1 to 4hr', '4hr to Day 1'], ylabel=r'$\Delta\_overlap',
                      alpha=0.5)

ax.set_title(group_title)
fig.savefig(os.path.join(pathname, group_title + ' Change in Overlap after Injection.pdf'))

# Stats are a bit weird - even though it looks like ANI guys start out super steep and then level off,
# Control guys do a bit of the same, though to a lesser extent
t_doverlap, pt_doverlap = sstats.ttest_rel(oratio_good[:, 0, :].reshape(-1) - oratio_good[:, 1, :].reshape(-1),
                        oratio_good[:, 1, :].reshape(-1) - oratio_good[:, 2, :].reshape(-1))

## Get pvals for between group comparisons...
# First run top two cells with each group, then run this cell

t_do_bw1, p_do_bw1 = sstats.ttest_ind(oratio_goodc[:, 0, :].reshape(-1) - oratio_goodc[:, 1, :].reshape(-1),
                                      oratio_gooda[:, 0, :].reshape(-1) - oratio_gooda[:, 1, :].reshape(-1))

t_do_bw2, p_do_bw2 = sstats.ttest_ind(oratio_goodc[:, 1, :].reshape(-1) - oratio_goodc[:, 2, :].reshape(-1),
                                      oratio_gooda[:, 1, :].reshape(-1) - oratio_gooda[:, 2, :].reshape(-1))


## Plot Number of Neurons active for each session

import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import cell_tracking as ct
import eraser_reference as err

group_title = 'Anisomycin'
if group_title is 'Control':
    mice = err.control_mice_good
elif group_title is 'Anisomycin':
    mice = err.ani_mice_good
days = [-2, -1, 4, 1, 2, 7]

arenas = ['Shock', 'Open']

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
nneurons = ct.get_group_num_neurons(mice, days=days, arenas=arenas)

fig, ax = ct.plot_num_neurons(nneurons)
ax.set_title(group_title)
fig.savefig(os.path.join(pathname, group_title + ' - ' + 'NumNeurons.pdf'))

fig2, ax2 = ct.plot_num_neurons(nneurons, normalize='1')
ax2.set_title(group_title)
fig2.savefig(os.path.join(pathname, group_title + ' - ' + 'NumNeuronsNorm.pdf'))

# Combine and plot
nneurons_comb = np.expand_dims(np.concatenate((nneurons[:, 0, :], nneurons[:, 1, :])), 1)
figc, axc = ct.plot_num_neurons(nneurons_comb)
axc.set_title(group_title + ' Combined')
figc.savefig(os.path.join(pathname, group_title + ' Combined - ' + 'NumNeurons.pdf'))

## Get stats for difference between all the above!
import scipy.stats as stats
import cell_tracking as ct
import eraser_reference as err
import matplotlib.pyplot as plt
import os as os
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
days = [-2, -1, 4, 1, 2, 7]
day_labels = ['-2', '-1', '4hr', '1', '2', '7']
arenas = ['Shock', 'Open']
norm_day = -1  # use -1 until you fix Marble20 day -2 data, then use -2

norm_sesh_ind = [days.index(i) for i in days if norm_day == i][0]
nneurons_c = ct.get_group_num_neurons(err.control_mice_good, days=days, arenas=arenas)
nnormc = ct.norm_num_neurons(nneurons_c, norm_sesh_ind)
nneurons_a = ct.get_group_num_neurons(err.ani_mice_good, days=days, arenas=arenas)
nnorma = ct.norm_num_neurons(nneurons_a, norm_sesh_ind)

## Do ANOVA for number neurons changing across all days
nmice, narena, ndays = nneurons_a.shape


# Friedman test - probably not enough samples for this to be legit
# Omit day -2 (index 0) for now due to nans from Marble20
stata, pfra = stats.friedmanchisquare(nneurons_a[:, :, 1].reshape(-1), nneurons_a[:, :, 2].reshape(-1),
                            nneurons_a[:, :, 3].reshape(-1), nneurons_a[:, :, 4].reshape(-1),
                            nneurons_a[:, :, 5].reshape(-1))


statc, pfrc = stats.friedmanchisquare(nneurons_c[:, :, 0].reshape(-1), nneurons_c[:, :, 1].reshape(-1),
                            nneurons_c[:, :, 2].reshape(-1), nneurons_c[:, :, 3].reshape(-1),
                            nneurons_c[:, :, 4].reshape(-1), nneurons_c[:, :, 5].reshape(-1))

# ANOVA - really should be a repeated measures ANOVA
stata, panovaa = stats.f_oneway(nneurons_a[:, :, 1].reshape(-1), nneurons_a[:, :, 2].reshape(-1),
                            nneurons_a[:, :, 3].reshape(-1), nneurons_a[:, :, 4].reshape(-1),
                            nneurons_a[:, :, 5].reshape(-1))
stata_nrm, panovaa_nrm = stats.f_oneway(nnorma[:, :, 1].reshape(-1),
                            nnorma[:, :, 3].reshape(-1), nnorma[:, :, 4].reshape(-1),
                            nnorma[:, :, 5].reshape(-1))
# Omit Marble 14 day 2 due to missing data
statc, panovac = stats.f_oneway(nneurons_c[:, :, 0].reshape(-1), nneurons_c[:, :, 1].reshape(-1),
                                nneurons_c[:, :, 2].reshape(-1), nneurons_c[:, :, 3].reshape(-1),
                                nneurons_c[[0, 1, 2, 3, 5], :, 4].reshape(-1),
                                nneurons_c[:, :, 5].reshape(-1))
statc_nrm, panovac_nrm = stats.f_oneway(nnormc[:, :, 0].reshape(-1),
                                        nnormc[:, :, 2].reshape(-1), nnormc[:, :, 3].reshape(-1),
                                        nnormc[[0, 1, 2, 3, 5], :, 4].reshape(-1),
                                        nnormc[:, :, 5].reshape(-1))
## Do post-hoc comparisons
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Remove marble20 since he has NaNs for day -1
nnorma_no20 =nnorma[[0, 1, 3], :, :]
groupsa = np.tile(np.asarray([1, 2, 3, 4, 5]), [3, 2, 1])

pha = pairwise_tukeyhsd(nnorma_no20[:, :, [0, 2, 3, 4, 5]].reshape(-1),
                        groupsa.reshape(-1))
print(pha)

# Remove marble20 since he has NaNs for day -1
groupsc = np.tile(np.asarray([1, 2, 3, 4, 5]), [5, 2, 1])
nnormc_no14 = nnormc[[0, 1, 2, 3, 5], :, :]
phc = pairwise_tukeyhsd(nnormc_no14[:, :, [0, 2, 3, 4, 5]].reshape(-1),
                        groupsc.reshape(-1))
print(phc)

## Check between shock and neutral arena within groups
tc_win = np.ones(len(days))*np.nan
pc_win = np.ones(len(days))*np.nan
ta_win = np.ones(len(days))*np.nan
pa_win = np.ones(len(days))*np.nan
t_bw = np.ones(len(days))*np.nan
p_bw = np.ones(len(days))*np.nan
trks_bw = np.ones(len(days))*np.nan
prks_bw = np.ones(len(days))*np.nan

# Get within group stats for each day - no differences!
for idd, day in enumerate(days):
    tc_win[idd], pc_win[idd] = stats.ttest_ind(nneurons_c[:, 0, idd],
                                               nneurons_c[:, 1, idd], nan_policy='omit')
    ta_win[idd], pa_win[idd] = stats.ttest_ind(nneurons_a[:, 0, idd],
                                               nneurons_a[:, 1, idd], nan_policy='omit')

# Get between group differences!
for idd, day in enumerate(days):
    # Independent t-test
    t_bw[idd], p_bw[idd] = stats.ttest_ind(nnormc.reshape((-1, len(days)))[:, idd],
                                            nnorma.reshape((-1, len(days)))[:, idd],
                                           nan_policy='omit')

    # Ranksum test
    trks_bw[idd], prks_bw[idd] = stats.ranksums(nnormc.reshape((-1, len(days)))[:, idd],
                                           nnorma.reshape((-1, len(days)))[:, idd])
p_bw1sided = p_bw/2

## Now plot them on top of one another!
nmicec, _, ndays = nnormc.shape
nmicea, _, _ = nnorma.shape
fig, ax = plt.subplots()
jitter = 0.05
ax.plot(np.matlib.repmat(np.arange(0, ndays), nmicec*2, 1) - jitter,
        nnormc.reshape((-1, ndays)), 'bo')
linec, = ax.plot(np.arange(0, ndays) - jitter, np.nanmean(nnormc.reshape((-1, ndays)),
                                                 axis=0), 'b-')
ax.plot(np.matlib.repmat(np.arange(0, ndays), nmicea*2, 1) + jitter,
        nnorma.reshape((-1, ndays)), 'ro')
linea, = ax.plot(np.arange(0, ndays)+ jitter, np.nanmean(nnorma.reshape((-1, ndays)),
                                                 axis=0), 'r-')
plt.legend((linec, linea), ('Control', 'Anisomycin'))
ax.set_xlabel('Day')
ax.set_xticks(np.arange(0, ndays))
ax.set_xticklabels(day_labels)
ax.set_ylabel('Normalized # Neurons')

fig.savefig(os.path.join(pathname, 'Between Group Normalized Neuron Plot.pdf'))

