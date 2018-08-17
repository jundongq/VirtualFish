import numpy as np 
import pandas as pd
import glob 
import os
import cv2
import matplotlib.pylab as plt



x_avg_observed, y_avg_observed, y_std_observed  = np.loadtxt('averageofObservedFishTrajectory.txt')
x_avg_virtual, y_avg_virtual, y_std_virtual = np.loadtxt('averageofVirtualFishTrajectory.txt')
# staring_pts = np.loadtxt('virtualFishStartingPts.txt')
# avarged_blocks = np.loadtxt('averge_blocks.txt')[::-1]

plt.figure(figsize=(16,8))
plt.errorbar(x_avg_observed, y_avg_observed, yerr=y_std_observed, c = 'k', fmt='o', alpha=0.6, label='Average Trend of Virtual Fish Trajectories')
plt.errorbar(x_avg_virtual, y_avg_virtual, yerr=y_std_virtual, c = 'g', fmt='x', alpha=0.6, label='Average Trend of Virtual Fish Trajectories')
# plt.scatter(x_avg_observed, y_avg_observed, c = 'k', marker = 'o', alpha=0.8, label='Average Trend of Observed Fish Trajectories')
# plt.scatter(x_avg_virtual, y_avg_virtual, c = 'g', marker = 'x', alpha=0.8, label='Average Trend of Virtual Fish Trajectories')
# plt.scatter(staring_pts[:,0], staring_pts[:,1], c = 'b', marker = '*', alpha=0.5, label='Starting Points of Virtual Fish')
# plt.imshow(avarged_blocks, cmap='gray', alpha = 0.6)
plt.xlim([0,2297])
plt.ylim([0,1284])
plt.xlabel("Length of field of view (FOV) in physical model (mm)")
plt.ylabel("Width of field of view (FOV) in physical model (mm)")
plt.axhline(y=70, c='k')
plt.axhline(y=1170, c='k')
plt.axhline(y=410, linestyle='--', c='k')
plt.legend(loc='upper right')
plt.show()
