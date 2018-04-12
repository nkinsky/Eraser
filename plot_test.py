import sys
sys.path.append(r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Python\Eraser')
sys.path.append(r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Python\FearReinstatement')
sys.path.append(r'C:\Users\kinsky.AD\Dropbox\Imaging Project\Python\FearReinstatement\Helpers')
import er_plot_functions as er
import os
os.chdir(r'E:\Eraser\Marble3\20180205_1_exposure')
os.getcwd()
freezing = er.detect_freezing(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd


import er_plot_functions as erplt
import matplotlib.pyplot as plt
dir1 = 'E:\\Eraser\\Marble3\\201080205_squareopen\\FreezeFrame2'
dir2 = 'E:\\Eraser\\Marble3\\20180205_1_exposure\\FreezeFrame'
fig, ax = plt.subplots(2,2)
erplt.plot_frame_and_traj(ax[0,0],dir1)
erplt.plot_frame_and_traj(ax[1,0],dir2)

posfile = os.path.join(dir1,'pos.csv')
vidfile = os.path.join(dir1,'20180319_openfeld_marble_7_video_crop.AVI')
pos3 = []
n = np.empty
n
n1 = np.array([[1,2,3],[4,5,6]])
n1[1,:]
n1.shape
with open(posfile,'r') as csvfile:
    # pos1 = list(csv.reader(csvfile))
    pos2 = csv.reader(csvfile)
    n = np.empty
    n1 = 0
    for row in pos2:
        print(row)
        pos3.append(row)
        n[n1,:] = row

test = pd.read_csv(posfile,header=None)
ax = test.T.plot(0,1,xlim = [20,130], ylim = [20,130])
ax.axis = 'off'
fig, axes = plt.subplots(2,2)
test.T.plot(0,1,xlim = [20,130], ylim = [20,130], ax = axes[0,1])
fig
import skvideo.io
vid = skvideo.io.vread(vidfile)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(vid[1])
ax[0,0].axis('off')
test.T.plot(0,1,xlim = [20,130], ylim = [20,130], ax = ax[0,0],legend = False)
fig
import sys
sys.path.append('C:\\Users\\Nat\\Documents\\BU\\Imaging\\Python\\FearReinstatement')
sys.path.append('C:\\Users\\Nat\\Documents\\BU\\Imaging\\Python\\FearReinstatement\\Helpers')

from session_directory import load_session_list
from mouse_sessions import make_session_list as msl
sd = msl('C:\\Users\\Nat\\Documents\\BU\\Imaging\\Working\\Eraser\\SessionDirectories')
x = [1,2,3,4,5]
x == 1
load_session_list('C:\\Users\\Nat\\Documents\\BU\\Imaging\\Working\\Eraser\\SessionDirectories')
