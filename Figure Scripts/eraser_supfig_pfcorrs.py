## Supplemental Figure for PF corrs
import Placefields as pf
import eraser_reference as err
import numpy as np
import er_plot_functions as erp
import os

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

## Get MI for all animals

mimean_discr = pf.load_all_mi(err.discriminators)
mimean_gen = pf.load_all_mi(err.generalizers)
mimean_ani = pf.load_all_mi(err.ani_mice_good)

data_cat = np.concatenate((mimean_discr.reshape(-1), mimean_gen.reshape(-1),
                           mimean_ani.reshape(-1)))

groups = np.concatenate((np.ones_like(mimean_discr.reshape(-1)), 2*np.ones_like(mimean_gen.reshape(-1)),
                           3*np.ones_like(mimean_ani.reshape(-1))))

fig, ax = erp.scatterbar(data_cat, groups)
ax.set_xlabel('Group (D, G, A)')
ax.set_ylabel('Mean MI (bits?)')

savefile= os.path.join(plot_dir, 'MI bw groups.pdf')

fig.savefig(savefile)


