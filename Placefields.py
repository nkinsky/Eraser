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
import er_plot_functions as er


def placefields(mouse, arena, day, list_dir='E:\Eraser\SessionDirectories'):
    """

    :param mouse:
    :param arena:
    :param day:
    :param list_dir:
    :return:
    """
    dir_use = get_dir(mouse, arena, day, list_dir)
    pos = er.get_pos(dir_use)
    _, _, pix2cm = er.get_conv_factors(arena)
    poscm = pos*pix2cm
    lims = [np.min(poscm,axis=1), np.max(poscm, axis=1)]


def makeoccmap(xcm, ycm, lims, good, isrunning, cmperbin):
    """
    Make Occupancy Map
    :param
        ax: matplotlib.pyplot axes
        vidfile: full path to video file
    :return:
    """

