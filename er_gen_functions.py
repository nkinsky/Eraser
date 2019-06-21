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
    # plt.axis('off')


def plot_tmap_sm(obj, ax_ind):
    """
    Plot smoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_sm[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_frames - 1
    obj.ax[ax_ind].axis('off')
    obj.ax[ax_ind].set_title(obj.titles[obj.current_position])


def plot_events_over_pos(obj, ax_ind):
    """
    Plot trajectory with calcium eve

    """
    psa_use = obj.PSAbool[obj.current_position, :]
    obj.ax[ax_ind].plot(obj.x, obj.y, 'k-')
    obj.ax[ax_ind].plot(obj.x[psa_use == 1], obj.y[psa_use == 1], 'r*')
    obj.ax[ax_ind].set_xlim(obj.traj_lims[0])
    obj.ax[ax_ind].set_ylim(obj.traj_lims[1])
    obj.last_position = obj.n_frames - 1
    obj.ax[ax_ind].set_title(obj.mouse + ' ' + obj.arena + ' Day ' + str(obj.day))
    # obj.ax[ax_ind].axis('off')
    # plt.axis('off')
    # print("sum psa_use = " + str(np.sum(psa_use)))


def plot_tmap_us2(obj, ax_ind):
    """
    Plot 2nd unsmoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_us2[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_frames - 1
    obj.ax[ax_ind].set_title('rho_spear = ' + str(round(obj.corrs_us[obj.current_position], 3)))
    # plt.axis('off')


def plot_tmap_sm2(obj, ax_ind):
    """
    Plot 2nd smoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_sm2[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_frames - 1
    obj.ax[ax_ind].axis('off')
    obj.ax[ax_ind].set_title('rho_spear = ' + str(round(obj.corrs_sm[obj.current_position], 3)))


def plot_events_over_pos2(obj, ax_ind):
    """
    Plot 2nd trajectory with calcium events

    """
    psa_use = obj.PSAbool2[obj.current_position, :]
    obj.ax[ax_ind].plot(obj.x2, obj.y2, 'k-')
    obj.ax[ax_ind].plot(obj.x2[psa_use == 1], obj.y2[psa_use == 1], 'r*')
    obj.last_position = obj.n_frames - 1
    obj.ax[ax_ind].set_title(obj.arena2 + ' Day ' + str(obj.day2))
    # plt.axis('off')
    # print("sum psa_use = " + str(np.sum(psa_use)))


if __name__ == '__main__':

    pass
