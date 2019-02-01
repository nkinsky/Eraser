import numpy as np
import scipy as sp
from er_plot_functions import get_all_freezing
# print("To use the freezing_stats function you must initialize a list of mice \n "
#       "into a variable specifying its cohort's name")
# mice= [input("Enter list of mice to initialize get group freezing: ")]

Control = ['Marble6', 'Marble7', 'Marble11', 'Marble12', 'Marble14',
           'Marble24', 'Marble26', 'Marble29']
Control_wsub = ['Marble3', 'Marble6', 'Marble7', 'Marble11', 'Marble12',
                'Marble14', 'Marble24', 'Marble26', 'Marble29']
ANI_mice = ['Marble17', 'Marble18', 'Marble19', 'Marble20', 'Marble25']


def get_group_freezing(mice):
    """create a list of days to facilitate array creation"""
    # Days = [dayminus1, day1]#  #set days to empty lists to set up for loop#
    dayminus1 = []
    day1 = []
    day2 = []
    day7 = []
    for mouse in mice:
        temp = get_all_freezing(mouse, [-1, 1, 2, 7], ["Shock"])
        dayminus1 += [temp[0, 0]]
        day1 += [temp[0, 1]]
        day2 += [temp[0, 2]]
        day7 += [temp[0, 3]]

    # print(str(mice))
    return dayminus1, day1, day2, day7


def rel_ttest(mice):
    """

    :param mice:
    :return:
    """
    # code here to get dayminus1 and day 1
    dayminus1, day1, _, _ = get_group_freezing(mice)
    RTTest = sp.stats.ttest_rel(dayminus1, day1, nan_policy='omit')
    return RTTest


def ind_ttest(control_mice, exp_mice):
    dayminus1_control, day1_control, day2_control, day7_control = get_group_freezing(control_mice)
    dayminus1_exp, day1_exp, day2_exp, day7_exp = get_group_freezing(exp_mice)

    # subtract out day minus freezing
    day1vbase_control = np.asarray(day1_control) - \
                        np.asarray(dayminus1_control)
    day1vbase_exp = np.asarray(day1_exp) - \
                    np.asarray(dayminus1_exp)
    # independent tttest for dayminus1vdayminus1?, day1vday1, day2vday2, day7vday7
    ITTest_dayminus1 = sp.stats.ttest_ind(dayminus1_control, dayminus1_exp,
                                          nan_policy='omit')
    ITTest_day1 = sp.stats.ttest_ind(day1_control, day1_exp,
                                     nan_policy='omit')
    ITTest_day2 = sp.stats.ttest_ind(day2_control, day2_exp,
                                     nan_policy='omit')
    ITTest_day7 = sp.stats.ttest_ind(day7_control, day7_exp,
                                     nan_policy='omit')
    ITTest_day1vbase = sp.stats.ttest_ind(day1vbase_control,
                                          day1vbase_exp, nan_policy='omit')
    return ITTest_dayminus1, ITTest_day1, ITTest_day2, \
           ITTest_day7, ITTest_day1vbase


# Debugging code - enter function you wish to debug here!
if __name__ == '__main__':
    rel_ttest(['Marble19', 'Marble20'])

    pass