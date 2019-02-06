## Get overlap ratios for each session to the first
import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import numpy as np

control_mice_good = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble24']
ani_mice_good = ['Marble17', 'Marble19', 'Marble25']

group_title = 'Anisomycin'
mice = ani_mice_good  # ['Marble11']
days = [-1, 4, 1, 2, 7]
arenas = ['Shock', 'Open']
oratio1 = np.ndarray((len(mice), len(days), len(arenas)))
oratio2 = np.ndarray((len(mice), len(days), len(arenas)))
oratioboth = np.ndarray((len(mice), len(days), len(arenas)))

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
plot_ind = True
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
for idm, mouse in enumerate(mice):
    for idd, day in enumerate(days):
        for ida, arena in enumerate(arenas):
            oratio1[idm, idd, ida], oratio2[idm, idd, ida], oratioboth[idm, idd, ida] = \
                pfs.get_overlap(mouse, 'Shock', -2, arena, day)

    if plot_ind:
        fig, ax = er.plot_overlaps(oratio1[idm, :, :])
        fig.savefig(os.path.join(pathname, 'Cell Overlap ' + group_title + '-' + mouse + '.pdf'))
        plt.close(fig)

fig, ax = er.plot_overlaps(oratioboth)
ax.set_title(group_title)

## Plot Number of Neurons active for each session

import placefield_stability as pfs
import er_plot_functions as er
import os
import matplotlib.pyplot as plt
import cell_tracking as ct

control_mice_good = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble24']
ani_mice_good = ['Marble17', 'Marble19', 'Marble25']

group_title = 'Control'
mice = control_mice_good  # ani_mice_good  # ['Marble11']
days = [-2, -1, 4, 1, 2, 7]
arenas = ['Shock', 'Open']
nneurons = np.ndarray((len(mice), len(days), len(arenas)))

# Get overlapping cell ratios for each day/arena using Shock day -2 as a reference
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder
for idm, mouse in enumerate(mice):
    for idd, day in enumerate(days):
        for ida, arena in enumerate(arenas):
            nneurons[idm, idd, ida] = ct.get_num_neurons(mouse, '', '', er_arena=arena,
                                                         er_day=day)


fig, ax = ct.plot_num_neurons(nneurons)
ax.set_title(group_title)
fig.savefig(os.path.join(pathname, group_title + ' - ' + 'NumNeurons.pdf'))