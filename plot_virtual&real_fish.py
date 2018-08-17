import numpy as np 
import pandas as pd
import glob 
import os
import cv2
import matplotlib.pylab as plt



x_avg_observed, y_avg_observed, y_std_observed  = np.loadtxt('averageofObservedFishTrajectory.txt')
x_avg_virtual, y_avg_virtual, y_std_virtual = np.loadtxt('averageofVirtualFishTrajectory.txt')


plt.figure(figsize=(16,8))
plt.errorbar(x_avg_observed, y_avg_observed, yerr=y_std_observed, c = 'k', fmt='o', alpha=0.6, label='Average Trend of Virtual Fish Trajectories')
plt.errorbar(x_avg_virtual, y_avg_virtual, yerr=y_std_virtual, c = 'g', fmt='x', alpha=0.6, label='Average Trend of Virtual Fish Trajectories')

plt.xlim([0,2297])
plt.ylim([0,1284])
plt.xlabel("Length of field of view (FOV) in physical model (mm)")
plt.ylabel("Width of field of view (FOV) in physical model (mm)")
plt.axhline(y=70, c='k')
plt.axhline(y=1170, c='k')
plt.axhline(y=410, linestyle='--', c='k')
plt.legend(loc='upper right')
plt.show()
