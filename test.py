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

from time import sleep
from tqdm import tqdm


def testpb():

    t = 0
    for i in tqdm(range(100)):
        t =+ 1
        sleep(0.01)

    return t
