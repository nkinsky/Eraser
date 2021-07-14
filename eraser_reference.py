from os import environ

# Marble14 should also be included once 10Hz SR is accounted for and Shock day 2 is run
control_mice = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                'Marble26', 'Marble27', 'Marble29']

control_mice_good = ['Marble06', 'Marble07', 'Marble11', 'Marble12', 'Marble14', 'Marble24',
                     'Marble27', 'Marble29']

ani_mice = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble21', 'Marble25']

ani_mice_good = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble25']  # 'Marble18 ?'

all_mice = control_mice.copy()
all_mice.extend(ani_mice)

all_mice_good = control_mice_good.copy()
all_mice_good.extend(ani_mice_good)

nonlearners = ["Marble06", "Marble11", "Marble29"]

learners = ["Marble07", "Marble12", "Marble24", "Marble27"]

plot_dir = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'  # Plotting folder

# from os import environ
# try:
#     comp_name = environ['COMPUTERNAME']
# except KeyError:  # Above does NOT work for Unix-based systems
#     from os import uname
#     comp_name = uname()[1]
#
# if comp_name == 'NATLAPTOP':
#     pathname = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'
# elif comp_name == 'RKC-HAS-WD-0005':
#     pathname = r'C:\Users\kinsky\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'
# elif comp_name == 'Evans computer':
#     pathname = 'fill in folder to plot to here evan'

## function to grab computer name and relevant path(s) for plotting/working
def get_comp_name():
    """Get computer name and path(s) to figure location and working directory"""

    working_dir = ''
    try:
        comp_name = environ['COMPUTERNAME']
    except KeyError:  # Above does NOT work for Unix-based systems
        from os import uname
        comp_name = uname()[1]

    if comp_name == 'NATLAPTOP':
        pathname = r'C:\Users\Nat\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'
        working_dir = r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser'
        session_dir = r'C:\Eraser\SessionDirectories'
    elif comp_name == 'RKC-HAS-WD-0005':  # not really used anymore
        pathname = r'C:\Users\kinsky\Dropbox\Imaging Project\Manuscripts\Eraser\Figures'
    elif comp_name == 'Nathaniels-MacBook-Air.local':
        pathname = '/Users/nkinsky/Dropbox/Imaging Project/Manuscripts/Eraser/Figures'
        working_dir = '/Users/nkinsky/Documents/BU/Working/Eraser'
        session_dir = '/Users/nkinsky/Documents/BU/Working/Eraser/SessionDirectories'
    elif comp_name == 'Evans computer':
        pathname = 'fill in folder to plot to here evan'

    return comp_name, working_dir, pathname, session_dir


# Designate plotting folder
comp_name, working_dir, pathname, session_dir = get_comp_name()

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
        o_xmin = [18, 15, 26, 22, 23, 15, 20]
        o_ymin = [18, 17, 12, 15, 16, 21, 8]
        s_range = [200, 185]
        s_xmin_use = 65
        s_ymin_use = 25
    elif mouse_name == 'Marble07':
        o_range = [105, 100]
        o_xmin = [23, 15, 14, 20, 19, 20, 23]
        o_ymin = [18, 9, 12, 9, 12, 8, 17]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 20
    elif mouse_name == 'Marble11':
        o_range = [105, 100]
        o_xmin = [19, 18, 24, 20, 21, 23, 24]
        o_ymin = [6, 15, 2, 5, 3, 2, 5]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 20
    elif mouse_name == 'Marble12':
        o_range = [105, 100]
        o_xmin = [16, 18, 18, 16, 20, 20, 18]
        o_ymin = [10, 10, 0, 8, 8, 3, 4]
        s_range = [200, 200]
        s_xmin_use = 70
        s_ymin_use = 20
    elif mouse_name == 'Marble14':
        o_range = [105, 100]
        o_xmin = [15, 15, 16, 16, 16, 17, 15]
        o_ymin = [12, 9, 6, 9, 5, 11, 14]
        s_range = [200, 200]
        s_xmin_use = 70
        s_ymin_use = 8
    elif mouse_name == 'Marble17':
        o_range = [105, 100]
        o_xmin = [14, 13, 12, 16, 13, 15, 11]
        o_ymin = [7, 10, 9, 10, 13, 7, 11]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble18':
        o_range = [105, 100]
        o_xmin = [9, 15, 15, 15, 17, 18, 15]
        o_ymin = [13, 9, 10, 4, 10, 7, 6]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble19':
        o_range = [105, 100]
        o_xmin = [16, 21, 13, 16, 17, 8, 11]
        o_ymin = [9, 10, 11, 10, 9, 5, 9]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble20':
        o_range = [105, 100]
        o_xmin = [12, 11, 12, 13, 10, 11, 15]
        o_ymin = [2, 9, 7, 7, 10, 7, 5]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble24':
        o_range = [105, 100]
        o_xmin = [16, 19, 16, 16, 11, 17, 13]
        o_ymin = [6, 1, 7, 4, 1, 4, 8]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble25':
        o_range = [105, 100]
        o_xmin = [16, 14, 17, 13, 13, 14, 16]
        o_ymin = [2, 4, 5, 2, 4, 2, 2]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble27':
        o_range = [105, 100]
        o_xmin = [18, 16, 14, 12, 13, 16, 16]
        o_ymin = [6, 2, 5, 7, 2, 0, 2]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8
    elif mouse_name == 'Marble29':
        o_range = [105, 100]
        o_xmin = [18, 16, 18, 16, 15, 15, 18]
        o_ymin = [6, 5, 4, 10, 2, 4, 4]
        s_range = [200, 200]
        s_xmin_use = 60
        s_ymin_use = 8

    s_xmin = np.ones(7) * s_xmin_use
    s_ymin = np.ones(7) * s_ymin_use

    return o_range, o_xmin, o_ymin, s_range, s_xmin, s_ymin