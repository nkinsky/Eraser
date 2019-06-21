# Marble14 should also be included once 10Hz SR is accounted for and Shock day 2 is run
control_mice = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                'Marble26', 'Marble27', 'Marble29']

control_mice_good = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                     'Marble27', 'Marble29']

ani_mice = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble21', 'Marble25']

ani_mice_good = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble25']  # 'Marble18 ?'

all_mice = control_mice.copy()
all_mice.append(ani_mice)

all_mice_good = control_mice_good.copy()
all_mice_good.append(ani_mice_good)

generalizers = ["Marble29", "Marble11", "Marble06"]

discriminators = ["Marble12", "Marble24", "Marble27", "Marble07"]


# Plotting folder
pathname = r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'


def grab_ax_lims(ax):
    """
    Gets xmin, xmax, ymin, and ymax for all axes specified in ax
    :param ax: flattened array of axes
    :return: xmin, ymin, xmax, ymax
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for a in ax:
        xmin.append(a.get_xlim()[0])
        ymin.append(a.get_ylim()[0])
        xmax.append(a.get_xlim()[1])
        ymax.append(a.get_ylim()[1])

    return xmin, ymin, xmax, ymax


def get_arena_lims(mouse_name):
    """
    Get arena limits for running aligned placefield sessions for eachmouse
    :param mouse_name:
    :return: o_range, o_xmin, o_ymin, s_range, s_xmin, s_ymin: minimums and of x and y position data
    and ranges [xrange, yrange] of data used to roughly align trajectories by eye and make sure the same


    """
    import numpy as np
    if mouse_name == 'Marble06':
        o_range = [105, 100]
        o_xmin = [18, 15, 25, 21, 22, 13, 15]
        o_ymin = [18, 18, 15, 18, 18, 22, 10]
        s_range = [200, 185]
        s_xmin_use = 65
        s_ymin_use = 25
    elif mouse_name == 'Marble07':
        o_range = [105, 100]
        o_xmin = [23, 15, 19, 12, 18, 20, 23]
        o_ymin = [18, 9, 9, 10, 10, 8, 14]
        s_range = [170, 185]
        s_xmin_use = 80
        s_ymin_use = 25
    # elif mouse_name == 'Marble11':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble12':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble14':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble17':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble18':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble19':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble20':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    elif mouse_name == 'Marble24':
        o_range = [105, 88]
        o_xmin = [18, 18, 15, 17, 13, 18, 15]
        o_ymin = [11, 5, 8, 10, 6, 8, 10]
        s_range = [200, 185]
        s_xmin_use = 60
        s_ymin_use = 10
    # elif mouse_name == 'Marble2'5:
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble27':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =
    # elif mouse_name == 'Marble29':
        # o_range = [105, 100]
        # o_xmin =
        # o_ymin =

    s_xmin = np.ones(7) * s_xmin_use
    s_ymin = np.ones(7) * s_ymin_use

    return o_range, o_xmin, o_ymin, s_range, s_xmin, s_ymin