# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:59:20 2018

@author: Nat Kinsky
"""

import pandas as pd
from os import path
from glob import glob
import numpy as np
import csv

# list file appendices for each session and mouse names
ff_append = ['*1_EXPOSURE.csv', '*2_EXPOSURE.csv', '*4_4hr.csv',
             '*5_REEXPOSURE.csv', '*6_REEXPOSURE.csv', '*7_week.csv']
mouse_names = ['GENERAL_1', 'GENERAL_2', 'GENERAL_3', 'GENERAL_4']
ff_dir = r'C:\Users\Nat\Documents\BU\Imaging\Working\Eraser\GEN_pilots\GEN_1'

ff_paths = [None]*len(ff_append)
for ida, names in enumerate(ff_append):
    ff_paths[ida] = glob(path.join(ff_dir, names))


def get_ff_freezing(mouse_name):
    """ Pulls freezing from FreezeFrame for all 6 sessions (excluding shock session) for the mouse specified"""

    # Find which mouse you entered
    mouse_bool = []
    for mouse in mouse_names:
        mouse_bool.append(mouse_name == mouse)

    # Loop through and get all freezing data for each 60 sec interval + avg freezing
    frz_avg = [None]*len(ff_paths)
    frz_by_min = []
    for idf, ff_path in enumerate(ff_paths):
        ff_freeze_data = pd.read_csv(ff_path[0])
        mouse_min_frz = ff_freeze_data.iloc[int(np.where(mouse_bool)[0])+2, 1:11]
        frz_avg[idf] = mouse_min_frz.mean(axis=0)
        np.append(frz_by_min, np.reshape(mouse_min_frz.values, [10, 1]))

    return frz_avg, frz_by_min

if __name__ == '__main__':
    get_ff_freezing('GENERAL_1')
    pass
