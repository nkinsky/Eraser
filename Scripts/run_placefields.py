"""Scripts to run placefields on all sessions, making sure they are aligned between sessions within an arena
"""

##
import er_plot_functions as erp
import helpers as hp
import matplotlib.pyplot as plt
import Placefields as pf
import numpy as np
import eraser_reference as err

# ## Here is simple code to run placefields without aligning between sessions for a given mouse
# # See cells below for code to align open field data (trickier than shock due to small movements
# # of arena between days)
# arena_use = 'Shock'
# mice_use = err.ani_mice
# open_day_des = [-2, -1, 4, 0, 1, 2, 7]  # keep as-is or code below will break
# nshuf = 1
# pix2cm = erp.get_conv_factors(arena_use)
# good_bool = np.ones((len(mice_use), len(open_day_des))) == 0  # pre-allocate tracking boolean
# for idm, mouse_use in enumerate(mice_use):
#     print('Running un-aligned ' + arena_use + ' PFs for ' + mouse_use)
#     for idd, day in enumerate(open_day_des):
#
#         try:
#             pf.placefields(mouse_use, arena_use, day, cmperbin=1, save_file='placefields_cm1_autolims.pkl',
#                            nshuf=nshuf)
#             good_bool[idm, idd] = True
#         except:
#             good_bool[idm, idd] = False
#             print('Error for ' + mouse_use + ' day ' + str(day))

# 1st plot all trajectories for a mouse and then manually align them with the same range of x/y values
mouse_use = 'Marble11'
day_des = [-2, -1, 4, 0, 1, 2, 7]  # keep this as-is or lots of code will break
arenas_use = ['Open', 'Shock']

# 1) Plot all trajectories
fig, ax = erp.plot_experiment_traj(mouse_use, arenas=arenas_use, day_des=day_des, plot_frame=False)
fig.set_size_inches(17.5, 2.3*len(arenas_use))
[erp.axis_on(a) for a in ax.reshape(-1)]

# 2) Specify range of x/y values (same for all session) and xmin, ymin for each open session
# Move around by hand to make occupancy match for each session
o_range = [105, 100]  # adjust to make maze cover middle 90% of limits
o_xmin = [22, 15, 20, 16, 17, 18, 20]  # leave these alone
o_ymin = [10, 13, 10, 10, 10, 8, 8]  # leave these alone
hp.set_all_lim_range(ax[0], o_range, o_xmin, o_ymin)

## 2.5) Set xmin, ymin, and range for shock session
# IMPORTANT NOTE: these should all be the same for all sessions! Camera is in the exact same place!
s_range = [225, 225]
s_xmin_use = 50
s_ymin_use = 10
s_xmin = np.ones(len(day_des))*s_xmin_use
s_ymin = np.ones(len(day_des))*s_ymin_use
hp.set_all_lim_range(ax[1], s_range, s_xmin, s_ymin)

## 2.3) After adjusting/aligning by hand, run function to pull-out o_xmin and o_ymin
xmin, ymin, _, _ = err.grab_ax_lims(ax[0])
o_xmin = np.round(xmin)
o_ymin = np.round(ymin)
hp.set_all_lim_range(ax[0], o_range, o_xmin, o_ymin)

# Print out for easy copying
[int(x) for x in np.round(o_xmin)]
[int(x) for x in np.round(o_ymin)]

## Check your work if already ran
try:
    o_range, o_xmin, o_ymin, s_range, s_xmin, s_ymin = err.get_arena_lims(mouse_use)
except:
    print('error')

hp.set_all_lim_range(ax[0], o_range, o_xmin, o_ymin)
hp.set_all_lim_range(ax[1], s_range, s_xmin, s_ymin)

## Run checks on above work here
# 3) Plot out data and check that it is well aligned by eye

# 4) Run all overlaid as last check
fig2, ax2 = erp.plot_trajectory_overlay(mouse_use, arenas=['Open'], day_des=day_des, xmin=o_xmin,
                                        ymin=o_ymin, xlim=[0, o_range[0]], ylim=[0, o_range[1]])

# This plots each one versus day -2 if the above is not helpful
for idd, day in enumerate(day_des[1:]):
    _, ax2 = erp.plot_trajectory_overlay(mouse_use, arenas=['Open'], day_des=[-2, day], xmin=[o_xmin[0], o_xmin[idd+1]],
                                         ymin=[o_ymin[0], o_ymin[idd+1]], xlim=[0, o_range[0]], ylim=[0, o_range[1]])
    ax2[0, 0].set_title('Day -2 vs ' + str(day))

# 5) When done save at bottom of this file for each mouse in commented code - will need to input
# when running Placefields.placefields to manually align data

## Run placefields on all open field data!
nshuf = 1
o_pix2cm = erp.get_conv_factors('Open')
o_range, o_xmin, o_ymin, _, _, _ = err.get_arena_lims(mouse_use)
good_bool = np.ones(len(day_des)) == 0  # pre-allocate tracking boolean
for idd, day in enumerate(day_des):
    xlims_use = np.asarray([o_xmin[idd], o_xmin[idd] + o_range[0]])*o_pix2cm
    ylims_use = np.asarray([o_ymin[idd], o_ymin[idd] + o_range[1]])*o_pix2cm
    try:
        pf.placefields(mouse_use, 'Open', day, cmperbin=1, lims_method=[[xlims_use[0], ylims_use[0]],
                       [xlims_use[1], ylims_use[1]]], save_file='placefields_cm1_manlims_1shuf.pkl',
                       nshuf=nshuf)
        good_bool[idd] = True
    except:
        print('Error for ' + mouse_use + ' day ' + str(day))
        good_bool[idd] = False

## Now run for all shock data!
nshuf = 1
o_pix2cm = erp.get_conv_factors('Open')
s_pix2cm = erp.get_conv_factors('Shock')
_, _, _, s_range, s_xmin, s_ymin = err.get_arena_lims(mouse_use)
good_bool = np.ones(len(day_des)) == 0  # pre-allocate tracking boolean
for idd, day in enumerate(day_des):
    xlims_use = np.asarray([s_xmin[idd], s_xmin[idd] + s_range[0]])*s_pix2cm
    ylims_use = np.asarray([s_ymin[idd], s_ymin[idd] + s_range[1]])*s_pix2cm
    try:
        pf.placefields(mouse_use, 'Shock', day, cmperbin=1, lims_method=[[xlims_use[0], ylims_use[0]],
                       [xlims_use[1], ylims_use[1]]], save_file='placefields_cm1_manlims_1shuf.pkl',
                       nshuf=nshuf)
        good_bool[idd] = True
    except:
        print('Error for ' + mouse_use + ' day ' + str(day))
        good_bool[idd] = False

## Check it on day -2 and day 2 sessions
dayn2s = pf.load_pf(mouse_use, 'Shock', -2, pf_file='placefields_cm1_manlims_1shuf.pkl')
dayn2s.pfscroll()
dayn2o = pf.load_pf(mouse_use, 'Open', -2, pf_file='placefields_cm1_manlims_1shuf.pkl')
dayn2o.pfscroll()

day2s = pf.load_pf(mouse_use, 'Shock', 2, pf_file='placefields_cm1_manlims_1shuf.pkl')
day2s.pfscroll()
day2o = pf.load_pf(mouse_use, 'Open', 2, pf_file='placefields_cm1_manlims_1shuf.pkl')
day2o.pfscroll()

## Finally run all mice with aligned data and 1000 shuffles!


## Verify everything is overlaid for each box - scale by xrange and yrange
fig, ax = plt.subplots()
for idd, day in enumerate([-2, -1, 4, 1, 2, 7]):
    temp = pf.load_pf(mouse_use, 'Shock', day, pf_file='placefields_cm1_manlims_1shuf.pkl')
    ax.plot(temp.xrun, temp.yrun)
ax.set_xlim([temp.xEdges.min(), temp.xEdges.max()])
ax.set_ylim([temp.yEdges.min(), temp.yEdges.max()])

# add in code here to plot open field overlaid data in plots to the right and below shock data,
# then adjust range until trajectories and whitespace are aligned more-or-less by eye


## Saved open field data limits for animals - might want to put into function

# 'Marble06'
# o_range = [105,100]
# o_xmin = [18, 15, 25, 21, 22, 13, 15]
# o_ymin = [18, 18, 15, 18, 18, 22, 10]
# s_range = [200, 185]
# s_xmin_use = 65
# s_ymin_use = 25

# mouse_name = 'Marble24'
# o_range = [105, 88]
# o_xmin = [18, 18, 15, 17, 13, 18, 15]
# o_ymin = [11, 5, 8, 10, 6, 8, 10]
# s_range = [200, 185]
# s_xmin_use = 60
# s_ymin_use = 10
