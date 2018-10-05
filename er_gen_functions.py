# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 09:36:01 2018

@author: Nat Kinsky
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_tmap_us(obj, ax_ind):
    """
    Plot unsmoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_us[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_frames - 1
    plt.axis('off')


def plot_tmap_sm(obj, ax_ind):
    """
    Plot smoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_sm[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_frames - 1
    plt.axis('off')


def plot_events_over_pos(obj, ax_ind):
    """
    Plot trajectory with calcium events overlaid
    :return:
    """
    psa_use = obj.PSAbool[obj.current_position, :]
    obj.ax[ax_ind].plot(obj.x, obj.y, 'k-')
    obj.ax[ax_ind].plot(obj.x[psa_use == 1], obj.y[psa_use == 1], 'r*')
    obj.last_position = obj.n_frames - 1
    plt.axis('off')
    # print("sum psa_use = " + str(np.sum(psa_use)))


if __name__ == '__main__':

    pass
