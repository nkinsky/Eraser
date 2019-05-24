"""Scripts to run placefields on all sessions, making sure they are aligned between sessions within an arena
"""
import er_plot_functions as erp
import helpers as hp
import matplotlib.pyplot as plt
import Placefields as pf
import numpy as np

## Open field: 1st plot all trajectories for a mouse and then manually align them with the same range of x/y values
mouse_use = 'Marble24'
open_day_des = [-2, -1, 4, 0, 1, 2, 7]

# 1) Plot all trajectories
fig, ax = erp.plot_experiment_traj(mouse_use, arenas=['Open'], day_des=open_day_des)
fig.set_size_inches(17.5, 2.3)

# 2) Specify range of x/y values (same for all session) and xmin, ymin for each session
range = [105, 88]
xmin = [18, 18, 15, 17, 13, 18, 15]
ymin = [11, 5, 8, 10, 6, 8, 10]

# 3) Plot out data and check that it is well aligned by eye
hp.set_all_lim_range(ax, range, xmin, ymin)

# 4) Run all overlaid as last check
fig2, ax2 = erp.plot_trajectory_overlay(mouse_use, arenas=['Open'], day_des=open_day_des, xmin=xmin,
                                        ymin=ymin, xlim=[0, range[0]], ylim=[0, range[1]])

# This plots each one versus day -2 if the above is not helpful
for idd, day in enumerate(open_day_des[1:]):
    _, ax2 = erp.plot_trajectory_overlay(mouse_use, arenas=['Open'], day_des=[-2, day], xmin=[xmin[0], xmin[idd+1]],
                                         ymin=[ymin[0], ymin[idd+1]], xlim=[0, range[0]], ylim=[0, range[1]])
    ax2[0, 0].set_title('Day -2 vs ' + str(day))

# 5) When done save at bottom of this file for each mouse in commented code - will need to input
# when running Placefields.placefields to manually align data

## Run placefields on all open field data!
nshuf = 1
pix2cm = erp.get_conv_factors('Open')
for idd, day in enumerate(open_day_des):
    xlims_use = np.asarray([xmin[idd], xmin[idd] + range[0]])*pix2cm
    ylims_use = np.asarray([ymin[idd], ymin[idd] + range[1]])*pix2cm
    pf.placefields(mouse_use, 'Open', day, cmperbin=1, lims_method=[[xlims_use[0], ylims_use[0]],
                   [xlims_use[1], ylims_use[1]]], save_file='placefields_cm1_manlims.pkl',
                   nshuf=nshuf)

## Verify everything is overlaid for shock box
fig, ax = plt.subplots()
for idd, day in enumerate([-2, -1, 4, 1, 2, 7]):
    temp = pf.load_pf(mouse_use, 'Shock', day, pf_file='placefields_cm1_manlims.pkl')
    ax.plot(temp.xrun, temp.yrun)
ax.set_xlim([temp.xEdges.min(), temp.xEdges.max()])
ax.set_ylim([temp.yEdges.min(), temp.yEdges.max()])

# add in code here to plot open field overlaid data in plots to the right and below shock data,
# then adjust range until trajectories and whitespace are aligned more-or-less by eye


## Saved open field data limits for animals - might want to put into function
# mouse_name = 'Marble24'
# range = [105, 88]
# xmin = [18, 18, 15, 17, 13, 18, 15]
# ymin = [11, 5, 8, 10, 6, 8, 10]
