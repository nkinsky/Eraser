## Placefield correlation figures/analysis

import eraser_reference as err
import placefield_stability as pfs
import er_plot_functions as erp
import numpy as np
from pickle import dump, load
from session_directory import find_eraser_directory as get_dir
import os
import matplotlib.pyplot as plt

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

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

## Plot best rotations for each mouse
for mouse in err.all_mice_good:
    fig, ax = plt.subplots(4, 5)
    fig.set_size_inches(18, 11)
    for ida, arena in enumerate(['Open', 'Shock']):
        for idd, day in enumerate([-1, 4, 1, 2, 7]):
            try:
                erp.pf_rot_plot(mouse, arena, -2, arena, day, ax=ax[ida*2, idd])
            except FileNotFoundError:
                print('FileNotFoundError for ' + mouse + ' ' + arena + ' day ' + str(-2) + ' to day ' + str(day))
                # ax[ida * 2, idd].set_visible(False)
    for ida, arena in enumerate(['Open', 'Shock']):
        for idd, day in enumerate([1, 2, 7]):
            try:
                erp.pf_rot_plot(mouse, arena, 4, arena, day, ax=ax[ida * 2 + 1, idd+2])
            except FileNotFoundError:
                print('FileNotFoundError for ' + mouse + ' ' + arena + ' day ' + str(-2) + ' to day ' + str(day))
                # ax[ida * 2 + 1, idd + 2].set_visible(False)
    ax[1, 0].set_visible(False)
    ax[1, 1].set_visible(False)
    ax[3, 0].set_visible(False)
    ax[3, 1].set_visible(False)
    savefile = os.path.join(plot_dir, mouse + ' PFrots simple.pdf')
    fig.savefig(savefile)
    plt.close(fig)


## Run through and plot 1-d PV corrs for all mice and save
mice = err.all_mice_good
nshuf = 1000
PVtype = 'both'
for mouse in mice:
    print('Running 1d corr plot for ' + mouse)
    save_file = os.path.join(plot_dir, mouse + '_PV1' + PVtype + ' corrs_nshuf' + str(nshuf) + '.pdf')
    fig, ax = erp.plot_PV1_simple(mouse, nshuf=nshuf, PVtype=PVtype)

    if mouse in err.ani_mice:
        label_use = 'ANI'
    elif mouse in err.control_mice:
        label_use = 'CTRL'

    ax.set_title(ax.get_title() + ' ' + label_use)
    fig.savefig(save_file)
    plt.close(fig)
## Workhorse code below - run before doing much of the above

## Run shuffled PV1 correlations for each session pair
nshuf = 1000
arenas = ['Shock', 'Open']
days = [-2, -1, 0, 4, 1, 2, 7]
for mouse in err.all_mice_good:
    # for arena1 in arenas:
    # arena2 = arena1
    arena1 = 'Shock'
    arena2 = 'Open'
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            if id1 <+ id2:
                try:
                    print('Running shuffled PV1 corrs for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' +
                          arena2 + ' day ' + str(day2))
                    pfs.PV1_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf=nshuf)
                except FileNotFoundError:
                    print('FileNotFoundError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' +
                          str(day2))
## Identify the best rotation for each correlation between mice
days = [-2, -1, 0, 4, 1, 2, 7]
for mouse in err.all_mice_good:
    # for arena1 in ['Shock', 'Open']:
    # arena2 = arena1
    arena1 = 'Open'
    arena2 = 'Shock'
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            if id1 <= id2:  # Only run for sessions forward in time
                try:
                    # Construct unique file save name
                    save_name = 'best_rot_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + str(day2) + '.pkl'
                    dir_use = get_dir(mouse, arena1, day1)
                    save_file = os.path.join(dir_use, save_name)

                    # Only run if file not already saved.
                    if not os.path.exists(save_file):
                        print('Running best rot analysis for ' + mouse + ' ' + arena1 + ' day ' + str(day1) +
                              ' to ' + arena2 + ' day ' + str(day2))
                        best_corr_mean, best_rot, corr_mean_all = pfs.get_best_rot(mouse, arena1, day1, arena2, day2)

                    else:
                        print('Skipping: file already exists for ' + mouse + ' ' + arena1 + ' day ' + str(day1) +
                              ' to ' + arena2 + ' day ' + str(day2))
                except IndexError:  #(FileNotFoundError, IndexError, ValueError):
                    print('IndexError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                except ValueError:
                    print('ValueError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                except FileNotFoundError:
                    print('FileNotFoundError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' +
                          str(day2))
## Get correlations between shuffled maps within arenas for all mice
days = [-2, -1, 0, 4, 1, 2, 7]
nshuf = 100
check = []
# Add in something to not run if save_file already exists!
for mouse in err.all_mice_good:
    # for arena in ['Shock', 'Open']:
    arena1 = 'Shock'
    arena2 = 'Open'
    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            dir_use = get_dir(mouse, arena1, day1)
            file_name = 'shuffle_map_mean_corrs_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + \
                        str(day2) + '_nshuf' + str(nshuf) + '.pkl'
            save_file = os.path.join(dir_use, file_name)
            if os.path.exists(save_file):
                check = check + [mouse, arena1, day1, arena2, day2, True]
            if id1 <= id2 and not os.path.exists(save_file):  # Only run for sessions forward in time
                try:
                    ShufMap = pfs.ShufMap(mouse, arena1=arena1, day1=day1, arena2=arena2, day2=day2, nshuf=nshuf)
                    ShufMap.get_shuffled_corrs()
                    ShufMap.save_data()
                    check = check + [mouse, arena1, day1, arena2, day2, True]
                except:  # FileNotFoundError:
                    print('Error in ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                    check = check + [mouse, arena1, day1, arena2, day2, False]

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

