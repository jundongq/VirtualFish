import os
import glob
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


def densityEstimation(data):
	"""
	use kde, gridsearchCV to estimate the pdf of give dataset.
	data:  array_like, shape (n_samples, n_features)
	"""
	data = data[:,np.newaxis] # shape (n_samples, n_features)
	params = {'bandwidth': np.logspace(-3, 0.1, 20), 'kernel':('gaussian', 'tophat')}
	grid = GridSearchCV(KernelDensity(), params)
	grid.fit(data)
	kde = grid.best_estimator_
	return kde



def trainKDEs(all_files_5Hz_file_path):
	### train data records
	all_files_5Hz = glob.glob(os.path.join(all_files_5Hz_file_path,'*.csv'))
	all_dfs = []
	for i, f in enumerate(all_files_5Hz):
		df = pd.read_csv(f)
		all_dfs.append(df)

	all_stepLengths_motionAngles = pd.concat(all_dfs)
	"""
	'Current_location_cx_mm', 'Current_location_cy_mm','currentDecisions', 'currentDecisonAngles', 
	'currentStepLength', 'previousDecisonAngles', 'previousSpeed', 'previousStepLength'
	"""

	all_stepLengths_motionAngles_groups = all_stepLengths_motionAngles.groupby('currentDecisions')


	### use train data to compute probability density function, for sampling purpose
	kde_fishDecisionAngles_0 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(0)['currentDecisonAngles'].values)
	kde_fishDecisionAngles_1 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(1)['currentDecisonAngles'].values)
	kde_fishDecisionAngles_2 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(2)['currentDecisonAngles'].values)
	kde_fishDecisionAngles_3 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(3)['currentDecisonAngles'].values)

	kde_currentStepLength_0 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(0)['currentStepLength'].values)
	kde_currentStepLength_1 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(1)['currentStepLength'].values)
	kde_currentStepLength_2 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(2)['currentStepLength'].values)
	kde_currentStepLength_3 = densityEstimation(all_stepLengths_motionAngles_groups.get_group(3)['currentStepLength'].values)

	return kde_fishDecisionAngles_0, kde_fishDecisionAngles_1, kde_fishDecisionAngles_2, kde_fishDecisionAngles_3, kde_currentStepLength_0, kde_currentStepLength_1, kde_currentStepLength_2, kde_currentStepLength_3
