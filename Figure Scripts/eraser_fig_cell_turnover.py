## Get overlap ratios for each session to the first
import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import numpy as np
import eraser_reference as err

group_title = 'Control'
mice = err.control_mice_good  # ['Marble11']
days = [-1, 4, 1, 2, 7]
arenas = ['Shock', 'Open']
oratio1 = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratio2 = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratioboth = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratiomin = np.ones((len(mice), len(days), len(arenas)))*np.nan
oratiomax = np.ones((len(mice), len(days), len(arenas)))*np.nan

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
plot_ind = True
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
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
        fig, ax = er.plot_overlaps(oratio1[idm, :, :])
        fig.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + '-' + mouse + '.pdf'))
        plt.close(fig)

fig, ax = er.plot_overlaps(oratioboth)
ax.set_title(group_title + ' Both normalized')
fig.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Both normalized.pdf'))
fig2, ax2 = er.plot_overlaps(oratiomax)
ax2.set_title(group_title + ' Max normalized')
fig2.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Max normalized.pdf'))
fig3, ax3 = er.plot_overlaps(oratiomin)
ax3.set_title(group_title + ' Min normalized')
fig3.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + ' Min normalized.pdf'))

## Plot Number of Neurons active for each session

import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import cell_tracking as ct
import eraser_reference as err

group_title = 'Anisomycin'
mice = err.ani_mice_good
days = [-2, -1, 4, 1, 2, 7]
arenas = ['Shock', 'Open']
nneurons = np.ones((len(mice), len(days), len(arenas)))*np.nan

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
for idm, mouse in enumerate(mice):
    for idd, day in enumerate(days):
        for ida, arena in enumerate(arenas):
            try:
                nneurons[idm, idd, ida] = ct.get_num_neurons(mouse, '', '', er_arena=arena,
                                                         er_day=day)
            except TypeError:
                print('Missing neural data file for ' + mouse + ' Day ' + str(day) + ' ' + arena)


fig, ax = ct.plot_num_neurons(nneurons)
ax.set_title(group_title)
fig.savefig(os.path.join(pathname, group_title + ' - ' + 'NumNeurons.pdf'))