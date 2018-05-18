# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 11:06:20 2018

@author: Nat Kinsky
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from os import path
import skvideo.io
from glob import glob
from session_directory import find_eraser_directory as get_dir
from er_plot_functions import get_pos

def MakeOccMap(xcm,ycm,lims,good,isrunning,cmperbin):
    """
    Make Occupancy Map
    :param
        ax: matplotlib.pyplot axes
        vidfile: full path to video file
    :return:
    """