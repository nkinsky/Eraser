## Placefield correlation figures/analysis

import eraser_reference as err

## Step through each mouse/day and construct confusion matrices
import eraser_reference as err
import placefield_stability as pfs
import numpy as np
import matplotlib.pyplot as plt
import os

# Variables to specify!
group_type = 'Anisomycin'  # 'Control'
mice = err.ani_mice_good  # err.control_mice_good
arena1 = 'Shock'
arena2 = 'Shock'
days = [-2, -1, 0, 4, 1, 2, 7]

# pre-allocate
ndays = len(days)
nmice = len(mice)
corr_sm_mean_all = np.ones((nmice, ndays, ndays))*np.nan

# Loop through each mouse and get mean correlations
for idm, mouse in enumerate(mice):
    _, corr_sm_mean_all[idm, :, :] = pfs.pf_corr_mean(mouse, arena1, arena2, days)

# Define groups for scatter plots
groups = np.ones_like(corr_sm_mean_all)*np.nan
groups[:, 0:2, 0:2] = 1  # 1 = before shock
groups[:, 4:7, 4:7] = 2  # 2 = after shock
groups[:, 0:2, 4:7] = 3  # 3 = before-v-after shock
groups[:, 0:2, 3] = 4  # 4 = before-v-STM
groups[:, 3, 4:7] = 5  # 5 = STM-v-LTM

# Plot corrs in scatterplot form
fig, ax = plt.subplots()
ax.scatter(groups.reshape(-1), corr_sm_mean_all.reshape(-1))
ax.set_xticks(np.arange(1, 6))
ax.set_xticklabels(['Before Shk', 'After Shk', 'Bef-Aft', 'Bef-STM', 'STM-Aft'])
ax.set_ylabel('Mean Spearmman Rho')
ax.set_title(group_type)
fig.savefig(os.path.join(err.pathname, 'PFcorrs ' + arena1 + ' v ' + arena2 + ' ' + group_type + '.pdf'))

# Plot corrs in confusion matrix
fig2, ax2 = plt.subplots()
ax2.imshow(np.nanmean(corr_sm_mean_all, axis=0))
ax2.set_xlim((0.5, ndays - 0.5))
ax2.set_ylim((ndays-1.5, -0.5))
ax2.set_xticklabels(['-2', '-1', '0', '4hr', '1', '2', '7'])
ax2.set_yticklabels([' ', '-2', '-1', '0', '4hr', '1', '2', '7'])
ax2.set_xlabel(arena2 + ' Day #')
ax2.set_ylabel(arena1 + ' Day #')
ax2.set_title(' Mean Spearman Rho: ' + group_type)
fig2.savefig(os.path.join(err.pathname, 'PFcorr Matrices ' + arena1 + ' v ' + arena2 + ' ' + group_type + '.pdf'))

