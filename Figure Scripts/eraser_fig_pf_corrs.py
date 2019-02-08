## Placefield correlation figures/analysis


## Step through each mouse/day and construct confusion matrices
import eraser_reference as err
import placefield_stability as pfs
import numpy as np
import matplotlib.pyplot as plt
import os

# Variables to specify!
# group_type = 'Control'  # 'Control'
# mice = err.control_mice_good  # err.control_mice_good
arena1 = 'Shock'
arena2 = 'Shock'
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

# Scatterplot for each group independently
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
                 ndays=ndays, )
pfs.plot_confmat(np.nanmean(ani_corr_sm_mean_all, axis=0), arena1, arena2, 'Anisomycin',
                 ndays=ndays)

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

