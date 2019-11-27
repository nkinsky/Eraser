## Supplemental Figure for PF corrs
import Placefields as pf
import eraser_reference as err
import numpy as np
import er_plot_functions as erp
import os
import matplotlib.pyplot as plt

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

## Get MI for all animals

mimean_discr = pf.load_all_mi(err.discriminators)
mimean_gen = pf.load_all_mi(err.generalizers)
mimean_ani = pf.load_all_mi(err.ani_mice_good)

data_cat = np.concatenate((mimean_discr.reshape(-1), mimean_gen.reshape(-1),
                           mimean_ani.reshape(-1)))

# Plot pooled data across arenas between groups
groups = np.concatenate((np.ones_like(mimean_discr.reshape(-1)), 2*np.ones_like(mimean_gen.reshape(-1)),
                           3*np.ones_like(mimean_ani.reshape(-1))))

fig, ax = erp.scatterbar(data_cat, groups)
ax.set_xlabel('Group (D, G, A)')
ax.set_ylabel('Mean MI (bits?)')

savefile1 = os.path.join(plot_dir, 'MI bw groups.pdf')
fig.savefig(savefile1)

# Now plot between arenas
fig2, ax2 = plt.subplots()
mi_neutral = np.concatenate((mimean_discr[:, 0, :].reshape(-1), mimean_gen[:, 0, :].reshape(-1),
                            mimean_ani[:, 0, :].reshape(-1)))
mi_shock = np.concatenate((mimean_discr[:, 1, :].reshape(-1), mimean_gen[:, 1, :].reshape(-1),
                          mimean_ani[:, 1, :].reshape(-1)))
arena_groups = np.concatenate((np.ones_like(mimean_discr[:, 0, :].reshape(-1)), 2*np.ones_like(mimean_gen[:, 0, :].reshape(-1)),
                              3*np.ones_like(mimean_ani[:, 0, :].reshape(-1))))

erp.scatterbar(mi_neutral, arena_groups, ax=ax2, color='b', offset=-0.125, data_label='Neutral',
               jitter=0.05)
erp.scatterbar(mi_shock, arena_groups, ax=ax2, color='r', offset=0.125, data_label='Shock',
               jitter=0.05)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['Discriminators', 'Generalizers', 'Anisomycin'])
plt.legend()

savefile2 = os.path.join(plot_dir, 'MI arena x groups.pdf')
fig2.savefig(savefile2)

