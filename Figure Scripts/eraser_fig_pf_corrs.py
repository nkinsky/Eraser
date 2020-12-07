## Placefield correlation figures/analysis

import eraser_reference as err
import placefield_stability as pfs
import Placefields as pf
import er_plot_functions as erp
from plot_helper import pretty_plot
import numpy as np
from pickle import dump, load
from session_directory import find_eraser_directory as get_dir
import os
import matplotlib.pyplot as plt
from glob import glob

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

fixed_reg = ['Marble06', 'Marble07', 'Marble12', 'Marble17', 'Marble18', 'Marble20', 'Marble25']
good_reg = ['Marble11', 'Marble14', 'Marble19', 'Marble24', 'Marble27', 'Marble29']

## Step through each mouse/day and construct confusion matrices

# Variables to specify!
# group_type = 'Control'  # 'Control'
# mice = err.control_mice_good  # err.control_mice_good
arena1 = 'Shock'
arena2 = 'Shock'
# arena1 = 'Open'
# arena2 = 'Open'
cmice = err.learners #err.control_mice_good
amice = err.ani_mice_good
days = [-2, -1, 0, 4, 1, 2, 7]
group_desig = 2  # 1 = include days 1,2, AND 7 in after shock group, 2 = include days 1 and 2 only
batch_map = False  #

ndays = len(days)
_, cont_corr_sm_mean_all = pfs.get_group_pf_corrs(cmice, arena1, arena2, days, batch_map_use=batch_map)
_, ani_corr_sm_mean_all = pfs.get_group_pf_corrs(amice, arena1, arena2, days, batch_map_use=batch_map)

# # pre-allocate

# nmicec = len(cmice)
# nmicea = len(amice)
# cont_corr_sm_mean_all = np.ones((nmicec, ndays, ndays))*np.nan
# ani_corr_sm_mean_all = np.ones((nmicea, ndays, ndays))*np.nan
#
# # Loop through each mouse and get mean correlations
# for idm, mouse in enumerate(cmice):
#     _, cont_corr_sm_mean_all[idm, :, :] = pfs.pf_corr_mean(mouse, arena1, arena2, days)
#
# for idm, mouse in enumerate(amice):
#     _, ani_corr_sm_mean_all[idm, :, :] = pfs.pf_corr_mean(mouse, arena1, arena2, days)

## Scatterplot for each group independently
pfs.plot_pfcorr_bygroup(cont_corr_sm_mean_all, arena1, arena2, 'Control', color='k',
                        group_desig=group_desig)
pfs.plot_pfcorr_bygroup(ani_corr_sm_mean_all, arena1, arena2, 'Anisomycin', color='g',
                        group_desig=group_desig)

# Combined scatterplots
figc, axc = pfs.plot_pfcorr_bygroup(cont_corr_sm_mean_all, arena1, arena2, '',
                                    color='k', offset=-0.1, save_fig=False, group_desig=group_desig)
pfs.plot_pfcorr_bygroup(ani_corr_sm_mean_all, arena1, arena2, 'Combined (green=Ani)',
                        color='g', offset=0.1, ax_use=axc, group_desig=group_desig, best_rot=False)

# Plot confusion matrices
pfs.plot_confmat(np.nanmean(cont_corr_sm_mean_all, axis=0), arena1, arena2, 'Control',
                 ndays=ndays)
pfs.plot_confmat(np.nanmean(ani_corr_sm_mean_all, axis=0), arena1, arena2, 'Anisomycin',
                 ndays=ndays)

## Plot best rotations for each mouse within arena
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

## Plot b/w arena best rotation plots on the same day for each mouse
for mouse in err.all_mice_good:
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(11, 5)
    for idd, day in enumerate([-2, -1, 4, 1, 2, 7]):
        try:
            erp.pf_rot_plot(mouse, 'Open', day, 'Shock', day, ax=ax.reshape(-1)[idd], nshuf=1000)
        except (FileNotFoundError, IndexError):
            print('FileNotFoundError or IndexError for ' + mouse + ' Open to Shock day ' + str(day))
        savefile = os.path.join(plot_dir, mouse + ' PFrots bw arenas simple.pdf')
        fig.savefig(savefile)
        plt.close(fig)

## Plot pf corrs at best rotation or PV1d for each group
amice = err.ani_mice_good
lmice = err.learners
nlmice = err.nonlearners
days = [-2, -1, 0, 4, 1, 2, 7]
group_desig = 2

type = 'PV1dall'  # 'PV1dboth' or 'PV1dall' or 'PF' are valid options
best_rot = True  # perform PFcorrs at best rotation between session if True, False = no rotation

learn_bestcorr_mean_all = []
nlearn_bestcorr_mean_all = []
ani_bestcorr_mean_all = []

for ida, arena in enumerate(['Open', 'Shock']):
    arena1 = arena
    arena2 = arena
    if type == 'PF':
        _, templ = pfs.get_group_pf_corrs(lmice, arena1, arena2, days, best_rot=best_rot)
        _, tempnl = pfs.get_group_pf_corrs(nlmice, arena1, arena2, days, best_rot=best_rot)
        _, tempa= pfs.get_group_pf_corrs(amice, arena1, arena2, days, best_rot=best_rot)
        prefix = 'PFcorrs'
    elif type == 'PV1dboth':
        _, templ = pfs.get_group_PV1d_corrs(lmice, arena1, arena2, days)
        _, tempnl = pfs.get_group_PV1d_corrs(nlmice, arena1, arena2, days)
        _, tempa = pfs.get_group_PV1d_corrs(amice, arena1, arena2, days)
        prefix = 'PV1dcorrs_both'
    elif type == 'PV1dall':
        templ, _ = pfs.get_group_PV1d_corrs(lmice, arena1, arena2, days)
        tempnl, _ = pfs.get_group_PV1d_corrs(nlmice, arena1, arena2, days)
        tempa, _ = pfs.get_group_PV1d_corrs(amice, arena1, arena2, days)
        prefix = 'PV1dcorrs_all'

    learn_bestcorr_mean_all.append(templ)
    nlearn_bestcorr_mean_all.append(tempnl)
    ani_bestcorr_mean_all.append(tempa)

    # Scatterplot for each group independently
    # pfs.plot_pfcorr_bygroup(disc_bestcorr_mean_all[ida], arena1, arena2, 'Discriminators', color='k',
    #                         group_desig=group_desig, best_rot=best_rot, save_fig=False, prefix=prefix)
    # pfs.plot_pfcorr_bygroup(gen_bestcorr_mean_all[ida], arena1, arena2, 'Generalizers', color='r',
    #                         group_desig=group_desig, best_rot=best_rot, save_fig=False, prefix=prefix)
    # pfs.plot_pfcorr_bygroup(ani_bestcorr_mean_all[ida],arena1, arena2, 'Anisomycin', color='g',
    #                             group_desig=group_desig, best_rot=best_rot, save_fig=False, prefix=prefix)

    #  Combined scatterplots
    figc, axc = pfs.plot_pfcorr_bygroup(learn_bestcorr_mean_all[ida], arena1, arena2, '', prefix=prefix,
                                        color='k', offset=0, save_fig=False, group_desig=group_desig)
    pfs.plot_pfcorr_bygroup(ani_bestcorr_mean_all[ida], arena1, arena2, '', prefix=prefix,
                            color='g', offset=0.1, ax_use=axc, group_desig=group_desig, save_fig=False)
    pfs.plot_pfcorr_bygroup(nlearn_bestcorr_mean_all[ida], arena1, arena2,
                            prefix + ' ' + arena1 + 'v' + arena2 + ' Combined (k=learn, b=n-learn, g=Ani) best_rot=' + str(best_rot), prefix=prefix,
                            color='b', offset=0.1, ax_use=axc, group_desig=group_desig, save_fig=True, best_rot=best_rot)

## Now do better combined plots
match_yaxis = True
match_ylims = [-0.3, 0.6]
lgroups, group_labels = pfs.get_time_groups(len(lmice), group_desig)
agroups, _ = pfs.get_time_groups(len(amice), group_desig)
nlgroups, _ = pfs.get_time_groups(len(nlmice), group_desig)
for idg, group in enumerate(np.unique(lgroups[~np.isnan(lgroups)]).tolist()):
    open_corrs1 = learn_bestcorr_mean_all[0][lgroups == group]
    shock_corrs1 = learn_bestcorr_mean_all[1][lgroups == group]
    open_corrs2 = ani_bestcorr_mean_all[0][agroups == group]
    shock_corrs2 = ani_bestcorr_mean_all[1][agroups == group]
    open_corrs3 = nlearn_bestcorr_mean_all[0][nlgroups == group]
    shock_corrs3 = nlearn_bestcorr_mean_all[1][nlgroups == group]


    fig, ax, pval, tstat = erp.pfcorr_compare(open_corrs1, shock_corrs1, open_corrs2, shock_corrs2, open_corrs3, shock_corrs3,
                   group_names=['Learn', 'Ani', 'Non-Learn'], xlabel=group_labels[idg], ylabel=prefix, xticklabels=['Open', 'Shock'])

    if type == 'PF':
        if not match_yaxis:
            savefile = os.path.join(plot_dir, prefix + ' 2x2 All groups best_rot=' + str(best_rot) + ' ' +
                                    group_labels[idg] +' .pdf')
        elif match_yaxis:
            ax[0].set_ylim(match_ylims)
            savefile = os.path.join(plot_dir, prefix + ' 2x2 All Groups best_rot=' + str(best_rot) + ' ' +
                                    group_labels[idg] +'_equalaxes.pdf')

        ax[0].set_title('best_rot=' + str(best_rot))
    else:
        savefile = os.path.join(plot_dir, prefix + ' 2x2 All Groups ' + group_labels[idg] + '.pdf')
    fig.savefig(savefile)

##  Plot day-by-day correlations
titles = ['Learners', 'Non-learners', 'Anisomycin']
fig, axuse = plt.subplots(3,1)
fig.set_size_inches([10.1, 9.2])
for idd, data in enumerate([learn_bestcorr_mean_all, nlearn_bestcorr_mean_all, ani_bestcorr_mean_all]):
    nmice = data[0].shape[0]
    gps, labels = pfs.get_seq_time_groups(nmice)
    erp.scatterbar(data[0][~np.isnan(gps)], gps[~np.isnan(gps)], data_label='Neutral', offset=-0.125,
                           jitter=0.05, color='k', ax=axuse[idd])
    erp.scatterbar(data[1][~np.isnan(gps)], gps[~np.isnan(gps)], data_label='Shock', offset=0.125,
                           jitter=0.05, color='r', ax=axuse[idd])
    axuse[idd].set_title(titles[idd] + ': ' + type)
    axuse[idd].set_xticks(np.unique(gps[~np.isnan(gps)]))
    axuse[idd].set_xticklabels(labels)
    if idd == 2:
        axuse[idd].legend()

## Get mean 95% CIs to put in plots manually for SfN, add into code above later. Only works for 2d corrs currently...

# preallocate
disc_allCI = np.ones((len(lmice), 3, 7, 7))*np.nan
disc_allCI_open = np.ones((len(lmice), 3, 7, 7))*np.nan
ani_allCI = np.ones((len(amice), 3, 7, 7))*np.nan
ani_allCI_open = np.ones((len(amice), 3, 7, 7))*np.nan

# Get CIs for all session-pairs/mice
nshuf = 100
for idm, mouse in enumerate(lmice):
    disc_allCI[idm, :, :, :] = pfs.get_all_CIshuf(mouse, 'Shock', 'Shock', nshuf=100)
    disc_allCI_open[idm, :, :, :] = pfs.get_all_CIshuf(mouse, 'Open', 'Open', nshuf=100)

for idm, mouse in enumerate(amice):
    ani_allCI[idm, :, :, :] = pfs.get_all_CIshuf(mouse, 'Shock', 'Shock', nshuf=100)
    ani_allCI_open[idm, :, :, :] = pfs.get_all_CIshuf(mouse, 'Open', 'Open', nshuf=100)

# Get mean CIs for all session-pairs at each stage after combining groups (vet this assumption later?)
CIcomb_shock = np.concatenate((disc_allCI, ani_allCI), 0)
CIcomb_open = np.concatenate((disc_allCI_open, ani_allCI_open), 0)

# Now estimate them by stage
groups, group_labels = pfs.get_time_groups(len(lmice) + len(amice), group_desig=group_desig)

unique_groups = np.unique(groups[~np.isnan(groups)])
ngroups = len(unique_groups)
CIshock_mean = np.ones((3, ngroups))*np.nan
CIopen_mean = np.ones((3, ngroups))*np.nan
for idg, group in enumerate(unique_groups):
    for j in range(3):
        CIshock_mean[j, idg] = np.nanmean(CIcomb_shock[:, j][groups == group])
        CIopen_mean[j, idg] = np.nanmean(CIcomb_open[:, j][groups == group])


## Compare best_rot=True to False in specified arena
amice = err.ani_mice_good
lmice = err.learners
days = [-2, -1, 0, 4, 1, 2, 7]
arena1 = 'Shock'
arena2 = 'Shock'
group_desig = 1

prefix = 'PFcorrs'
_, disc_corrs_norot = pfs.get_group_pf_corrs(lmice, arena1, arena2, days, best_rot=False)
_, disc_corrs_bestrot = pfs.get_group_pf_corrs(lmice, arena1, arena2, days, best_rot=True)
_, ani_corrs_norot = pfs.get_group_pf_corrs(amice, arena1, arena2, days, best_rot=False)
_, ani_corrs_bestrot = pfs.get_group_pf_corrs(amice, arena1, arena2, days, best_rot=True)

# Set up plots
match_yaxis = True
match_ylims = [-0.3, 0.6]
dgroups, group_labels = pfs.get_time_groups(len(lmice), group_desig)
agroups, _ = pfs.get_time_groups(len(amice), group_desig)

# Plot them all out
for idg, group in enumerate(np.unique(dgroups[~np.isnan(dgroups)]).tolist()):
    disc_corrs_norot_use = disc_corrs_norot[dgroups == group]
    disc_corrs_bestrot_use = disc_corrs_bestrot[dgroups == group]
    ani_corrs_norot_use = ani_corrs_norot[agroups == group]
    ani_corrs_bestrot_use = ani_corrs_bestrot[agroups == group]

    fig, ax, pval, tstat = erp.pfcorr_compare(disc_corrs_norot_use, disc_corrs_bestrot_use, ani_corrs_norot_use,
                                              ani_corrs_bestrot_use, colors=['k', 'g'], group_names=['Discr', 'Ani'],
                                              xlabel=group_labels[idg], ylabel=prefix, xticklabels=['no rot', 'best rot'])

    if not match_yaxis:
        savefile = os.path.join(plot_dir, prefix + ' ' + arena1 + 'v' + arena2 + ' 2x2 Discr and Ani no v best rot ' + group_labels[idg] + '.pdf')
    elif match_yaxis:
        ax[0].set_ylim(match_ylims)
        savefile = os.path.join(plot_dir, prefix + ' ' + arena1 + 'v' + arena2 + ' 2x2 Discr and Ani no v best rot ' + group_labels[idg] + '_equalaxes.pdf')

    ax[0].set_title('No rot v Best rot')
    fig.savefig(savefile)

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

## For mice with fixed registrations move all files to "archive" folders
arenas = ['Shock', 'Open']
days = [-2, -1, 0, 4, 1, 2, 7]
name_append = '_bad4'  # super ocd tracking of # times you've had to redo stuff _2 = 2nd, _87 = 87th, etc.

# IMPORTANT - comment out files you don't want to move in code below!
for mouse in fixed_reg:
    for arena1 in arenas:
        for arena2 in arenas:
            for id1, day1 in enumerate(days):
                for id2, day2 in enumerate(days):
                    if id1 <= id2 and arena1 != arena2 or id1 < id2 and arena1 == arena2:
                        dir_use = get_dir(mouse, arena1, day1)
                        archive_dir = os.path.join(dir_use, 'rot_archive')
                        if not os.path.isdir(archive_dir):
                            try:
                                os.mkdir(archive_dir)
                            except FileNotFoundError:
                                print('Error for ' + mouse + ' ' + arena1 + ' day ' + str(day1)
                                      + ' to ' + arena2 + ' day ' + str(day2))
                        # files_move = glob(os.path.join(dir_use, 'shuffle_map_mean*nshuf1000.pkl'))
                        files_move = glob(os.path.join(dir_use, 'best_rot*.pkl'))
                        # files_move.extend(glob(os.path.join(dir_use, 'PV1shuf*nshuf_1000.pkl')))
                        # files_move.extend(glob(os.path.join(dir_use, 'shuffle_map_mean*nshuf100.pkl')))
                        for file in files_move:
                            try:
                                _, f = os.path.split(file)
                                os.rename(file, os.path.join(archive_dir, f[0:-4] + name_append + '.pkl'))
                            except FileNotFoundError:
                                print('Error for ' + mouse + ' ' + arena1 + ' day ' + str(day1)
                                      + ' to ' + arena2 + ' day ' + str(day2))


## Run shuffled PV1 correlations for each session pair
nshuf = 1000
arenas = ['Shock', 'Open']
days = [-2, -1, 0, 4, 1, 2, 7]
for mouse in fixed_reg:
    # # for arena1 in arenas:


    # # arena2 = arena1
    # arena1 = 'Open'
    # arena2 = 'Shock'
    for arena1 in arenas:
        for arena2 in arenas:
            for id1, day1 in enumerate(days):
                for id2, day2 in enumerate(days):
                    if id1 <= id2 and arena1 != arena2 or id1 < id2 and arena1 == arena2:
                        try:
                            print('Running shuffled PV1 corrs for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' +
                                  arena2 + ' day ' + str(day2))
                            pfs.PV1_shuf_corrs(mouse, arena1, day1, arena2, day2, nshuf=nshuf)
                        except FileNotFoundError:
                            print('FileNotFoundError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' +
                                  str(day2))
## Identify the best rotation for each correlation between mice
days = [-2, -1, 0, 4, 1, 2, 7]
arenas = ['Open', 'Open']
batch_map = True


for mouse in err.all_mice_good:

    arena1 = arenas[0]
    arena2 = arenas[1]

    for id1, day1 in enumerate(days):
        for id2, day2 in enumerate(days):
            if id1 <= id2 and arena1 != arena2 or id1 < id2 and arena1 == arena2:
                try:
                    # Construct unique file save name
                    save_name = 'best_rot_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + str(day2) + \
                                '_batch_map=' + str(batch_map) + '.pkl'
                    dir_use = get_dir(mouse, arena1, day1)
                    save_file = os.path.join(dir_use, save_name)

                    # Only run if file not already saved.
                    if not os.path.exists(save_file):
                        print('Running best rot analysis for ' + mouse + ' ' + arena1 + ' day ' + str(day1) +
                              ' to ' + arena2 + ' day ' + str(day2))
                        best_corr_mean, best_rot, corr_mean_all = pfs.get_best_rot(mouse, arena1, day1, arena2, day2,
                                                                                   batch_map_use=batch_map)

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
                except TypeError:
                    print('TypeError for ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' +
                          arena2 + ' day ' + str(day2))
## Get correlations between shuffled maps within/between arenas for all mice
arenas = ['Open', 'Shock']
days = [-2, -1, 0, 4, 1, 2, 7]
nshuf = 1000
check = []
# Add in something to not run if save_file already exists!
for mouse in fixed_reg:
    # # for arena in ['Shock', 'Open']:
    # arena1 = 'Open'
    # arena2 = 'Shock'
    for arena1 in arenas:
        for arena2 in arenas:
            for id1, day1 in enumerate(days):
                for id2, day2 in enumerate(days):
                    dir_use = get_dir(mouse, arena1, day1)
                    file_name = 'shuffle_map_mean_corrs_' + arena1 + 'day' + str(day1) + '_' + arena2 + 'day' + \
                                str(day2) + '_nshuf' + str(nshuf) + '.pkl'
                    save_file = os.path.join(dir_use, file_name)
                    if id1 <= id2 and arena1 != arena2 or id1 < id2 and arena1 == arena2 and not os.path.exists(save_file):  # Only run for sessions forward in time
                        try:
                            ShufMap = pfs.ShufMap(mouse, arena1=arena1, day1=day1, arena2=arena2, day2=day2, nshuf=nshuf)
                            if not os.path.exists(ShufMap.save_file):
                                ShufMap.get_shuffled_corrs()
                                ShufMap.save_data()
                                check.append([mouse, arena1, day1, arena2, day2, True])
                            else:
                                print('Previously saved data for ' + mouse + ' ' + arena1 + ' day'  + str(day1) + ' to '
                                      + arena2 + ' day ' + str(day2) + ": Skipping.")
                                check.append([mouse, arena1, day1, arena2, day2, True])
                        except:  # FileNotFoundError:
                            print('Error in ' + mouse + ' ' + arena1 + ' day ' + str(day1) + ' to ' + arena2 + ' day ' + str(day2))
                            check.append([mouse, arena1, day1, arena2, day2, False])

## Get place-field correlation histograms at no rotation versus best rotation for all mice/arenas for comparison purposes
# Note that between arena plots aren't as useful. Better is to do each day versus itself and put on the same plot...
smooth = True
for mouse in err.all_mice_good:
    for arena1, arena2 in zip(['Open', 'Shock'], ['Open', 'Shock']):
        rotfigs, _ = pfs.compare_pf_at_bestrot(mouse, arena1=arena1, arena2=arena2, smooth=smooth)
        [fig.savefig(os.path.join(err.plot_dir, rot_text + ' rot PF corrs ' + mouse + ' ' + arena1 + 'v' + arena2 +
                                     ' smooth=' + str(smooth)) + '.pdf') for fig, rot_text in zip(rotfigs, ['No', 'Best'])]
        plt.close('all')


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

