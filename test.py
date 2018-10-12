# string = 'Hydrogen'
# for index, letter in enumerate(string):
#     print((letter,index))
#
# import numpy as np
# x = np.array([1,2,3])
#
# import matplotlib.pyplot as plt
# # %matplotlib inline
# y = np.linspace(0,20,500)
# plt.plot(y,np.cos(y),'b',y,np.sin(y),'r')
# plt.plot(np.sin(y),'r')
# plt.show()
#
# # %% Test comment
# y = np.linspace(0,20,500)
# plt.plot(y,np.cos(y),'b',y,np.sin(y),'r')
# plt.show()
#
# import sys
# sys.path

## Code to run placefields on all sessions for mouse entered below
import Placefields as pf

mouse = 'Marble07'
arenas = ['Open', 'Shock']
days = [-2, -1, 0, 4, 1, 2, 7]
for arena in arenas:
    for day in days:
        print('Processing ' + arena + ' day ' + str(day))
        try:
            pf.placefields(mouse, arena, day, list_dir=r'C:\Eraser\SessionDirectories',
                       nshuf=1000)
        except:
            print(['Unknown error processing session'])


# from tqdm import tqdm
# from progressbar import ProgressBar
# from time import sleep
#
# for i in tqdm(range(1000)):
#     sleep(0.01)