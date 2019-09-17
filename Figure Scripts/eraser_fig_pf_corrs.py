## Placefield correlation figures/analysis

import eraser_reference as err
import placefield_stability as pfs
import numpy as np
from pickle import dump, load
from session_directory import find_eraser_directory as get_dir
import os
import matplotlib.pyplot as plt
import os

## Step through each mouse/day and construct confusion matrices

# Variables to specify!
# group_type = 'Control'  # 'Control'
# mice = err.control_mice_good  # err.control_mice_good
# arena1 = 'Shock'
# arena2 = 'Shock'
arena1 = 'Open'
arena2 = 'Open'
cmice = err.control_mice_good
amice = err.ani_mice_good
days = [-2, -1, 0, 4, 1, 2, 7]
group_desig = 2  # 1 = include days 1,2, AND 7 in after shock group, 2 = include days 1 and 2 only

# pre-allocate
ndays = len(days)
nmicec = len(cmice)
nmicea = len(amice)
cont_corr_sm_mean_all = np.ones((nmicec, ndays, ndays))*np.nan
ani_corr_sm_mean_all = np.ones((nmicea, ndays, ndays))*np.nan

# Loop through each mouse and get mean correlations
for idm, mouse in enumerate(cmice):
    _, cont_corr_sm_mean_all[idm, :, :] = pfs.pf_corr_mean(mouse, arena1, arena2, days)

for idm, mouse in enumerate(amice):
    _, ani_corr_sm_mean_all[idm, :, :] = pfs.pf_corr_mean(mouse, arena1, arena2, days)

## Scatterplot for each group independently
pfs.plot_pfcorr_bygroup(cont_corr_sm_mean_all, arena1, arena2, 'Control', color='k',
                        group_desig=group_desig)
pfs.plot_pfcorr_bygroup(ani_corr_sm_mean_all, arena1, arena2, 'Anisomycin', color='g',
                        group_desig=group_desig)

# Combined scatterplots
figc, axc = pfs.plot_pfcorr_bygroup(cont_corr_sm_mean_all, arena1, arena2, '',
                                    color='k', offset=-0.1, save_fig=False, group_desig=group_desig)
pfs.plot_pfcorr_bygroup(ani_corr_sm_mean_all, arena1, arena2, 'Combined (green=Ani)',
                        color='g', offset=0.1, ax_use=axc, group_desig=group_desig)

# Plot confusion matrices
pfs.plot_confmat(np.nanmean(cont_corr_sm_mean_all, axis=0), arena1, arena2, 'Control',
                 ndays=ndays)
pfs.plot_confmat(np.nanmean(ani_corr_sm_mean_all, axis=0), arena1, arena2, 'Anisomycin',
                 ndays=ndays)

## Workhorse code below - run before doing much of the above to save shuffled map corrs and
# id best rotations
## Identify the best rotation for each correlation between mice
days = [-2, -1, 0, 4, 1, 2, 7]
for mouse in ['Marble07']:  #err.all_mice_good:
    for arena in ['Shock', 'Open']:
        for id1, day1 in enumerate(days):
            for id2, day2 in enumerate(days):
                if id1 < id2:  # Only run for sessions forward in time
                    print('Running best rot analysis for ' + mouse + ' ' + arena + ' day ' + str(day1) + ' to day ' +
                          str(day2))
                    best_corr_mean, best_rot, corr_mean_all = pfs.get_best_rot(mouse, arena, day1, arena, day2)
                    save_file = 'best_rot_' + arena + 'day' + str(day1) + '_' + arena + 'day' + str(day2) + '.pkl'
                    dump([['Mouse','Arena', 'day1', 'day2', 'best_corr_mean[un-smoothed, smoothed]', 'best_rot[un-smoothed, smoothed]',
                           'corr_mean_all[un-smoothed, smoothed]'], [mouse, arena, day1, day2, best_corr_mean, best_rot, corr_mean_all]],
                         open(save_file, "wb"))

## Get correlations between shuffled maps within arenas for all mice
days = [-2, -1, 0, 4, 1, 2, 7]
nshuf = 100
check = []
# Add in something to not run if save_file already exists!
for mouse in err.all_mice_good:
    for arena in ['Shock', 'Open']:
        for id1, day1 in enumerate(days):
            for id2, day2 in enumerate(days):
                dir_use = get_dir(mouse, arena, day1)
                file_name = 'shuffle_map_mean_corrs_' + arena + 'day' + str(day1) + '_' + arena + 'day' + \
                            str(day2) + '_nshuf' + str(nshuf) + '.pkl'
                save_file = os.path.join(dir_use, file_name)
                if os.path.exists(save_file):
                    check = check + [mouse, arena, day1, day2, True]
                if id1 < id2 and not os.path.exists(save_file):  # Only run for sessions forward in time
                    try:
                        ShufMap = pfs.ShufMap(mouse, arena1=arena, day1=day1, arena2=arena, day2=day2, nshuf=nshuf)
                        ShufMap.get_shuffled_corrs()
                        ShufMap.save_data()
                        check = check + [mouse, arena, day1, day2, True]
                    except:  # FileNotFoundError:
                        print('Error in ' + mouse + ' ' + arena + ' day ' + str(day1) + ' to day ' + str(day2))
                        check = check + [mouse, arena, day1, day2, False]

##
# Define groups for scatter plots
# groups = np.ones_like(corr_sm_mean_all)*np.nan
# groups[:, 0:2, 0:2] = 1  # 1 = before shock
# groups[:, 4:7, 4:7] = 2  # 2 = after shock
# groups[:, 0:2, 4:7] = 3  # 3 = before-v-after shock
# groups[:, 0:2, 3] = 4  # 4 = before-v-STM
# groups[:, 3, 4:7] = 5  # 5 = STM-v-LTM
#
# # Plot corrs in scatterplot form
# fig, ax = plt.subplots()
# ax.scatter(groups.reshape(-1), corr_sm_mean_all.reshape(-1))
# ax.set_xticks(np.arange(1, 6))
# ax.set_xticklabels(['Before Shk', 'After Shk', 'Bef-Aft', 'Bef-STM', 'STM-Aft'])
# ax.set_ylabel('Mean Spearmman Rho')
# ax.set_title(group_type)
# unique_groups = np.unique(groups[~np.isnan(groups)])
# corr_means = []
# for group_num in unique_groups:
#     corr_means.append(np.nanmean(corr_sm_mean_all[groups == group_num]))
# ax.plot(unique_groups, corr_means, 'b-')
# fig.savefig(os.path.join(err.pathname, 'PFcorrs ' + arena1 + ' v ' + arena2 + ' ' + group_type + '.pdf'))

# Plot corrs in confusion matrix
# fig2, ax2 = plt.subplots()
# ax2.imshow(np.nanmean(corr_sm_mean_all, axis=0))
# ax2.set_xlim((0.5, ndays - 0.5))
# ax2.set_ylim((ndays-1.5, -0.5))
# ax2.set_xticklabels(['-2', '-1', '0', '4hr', '1', '2', '7'])
# ax2.set_yticklabels([' ', '-2', '-1', '0', '4hr', '1', '2', '7'])
# ax2.set_xlabel(arena2 + ' Day #')
# ax2.set_ylabel(arena1 + ' Day #')
# ax2.set_title(' Mean Spearman Rho: ' + group_type)
# fig2.savefig(os.path.join(err.pathname, 'PFcorr Matrices ' + arena1 + ' v ' + arena2 + ' ' + group_type + '.pdf'))

