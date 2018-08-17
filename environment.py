"""
This script is to do a time averaged over 4 time steps on vel, vorticity, and tke, 05/08/2018
Add swirl and strain rate into the features, 05/24/2018
Add acceleration of u and v into features, 06/01/2018
"""

import numpy as np 
import pandas as pd 
import glob
import os
import argparse
import shutil
import h5py

def u_v(vel_csv_file):

	"""input as velocity data exported from Streams; output as two matrices containing velocity field of each image """

	# vel_csv_file is the output from Streams. The 1st row needs to be skipped
	velocity = pd.read_csv(vel_csv_file, index_col = [0], skiprows=[0], na_values =' NaN')

	# convert all NaN to 0s.
	velocity.fillna(0, inplace=True)

	# total number of rows with valid data for each velocity component
	total_rows = nb_matrix * nb_y_rows

	# initiate numpy arrays to store velocity data from velocity dataframe
	u_velocity = np.zeros((nb_matrix, nb_y_rows, nb_x_cols))
	v_velocity = np.zeros((nb_matrix, nb_y_rows, nb_x_cols))

	for i in range(nb_matrix):
		u_velocity[i] = velocity.values[i*nb_y_rows:(i+1)*nb_y_rows , :]
		v_velocity[i] = velocity.values[total_rows+i*nb_y_rows:total_rows+(i+1)*nb_y_rows, :]

	return u_velocity, v_velocity


def vor_tke(vor_tke_csv_file):

	"""input as vorticity/tke data exported from Streams; output as one matrices containing vorticity/tke field of each image """

	# vorticity/tke_csv_file is the output from Streams. The 1st row needs to be skipped
	df_full = pd.read_csv(vor_tke_csv_file, index_col = [0], skiprows=[0], na_values =' NaN')

	# convert all NaN to 0s.
	df_full.fillna(0, inplace=True)

	# total number of rows with valid data for each velocity component
	total_rows = nb_matrix * nb_y_rows

	# initiate numpy arrays to store velocity data from velocity dataframe
	df_full_ = np.zeros((nb_matrix, nb_y_rows, nb_x_cols))

	for i in range(nb_matrix):
		df_full_[i] = df_full.values[i*nb_y_rows:(i+1)*nb_y_rows , :]


	return df_full_


def zeros_pad(a, patch_width, patch_height):

	"""
	input::
	a: a 2D matrix
	patch_width: the width of a fish sensory area
	patch_height: the height of a fish sensory area
	
	output:
	a_pad: matrix a with zero padding
	"""
	a_pad = np.pad(a, [(patch_height/2, patch_height/2), (patch_width/2, patch_width/2)], mode='constant', constant_values=0)
	return a_pad

def average_feature(u, avg_nb_decisions, avg_steps):
	"""
	Input:
	u: a tensor in format of (nb_matrix, nb_y_rows, nb_x_cols), which needs to be averaged over avg_steps consecutive steps
	avg_nb_decisions: total number of decisions at 5Hz
	avg_steps: number of steps for averaging.

	Output:
	u_avg: a tenor in format of (avg_nb_decisions, nb_y_rows, nb_x_cols)
	"""
	u_avg = np.zeros((avg_nb_decisions, nb_y_rows, nb_x_cols))
	for i in range(avg_nb_decisions):
		u_avg[i,:,:] = u[i*avg_steps:(i+1)*avg_steps,:, :].sum(0)/(u[i*avg_steps:(i+1)*avg_steps,:, :]!=0).sum(0)
	return u_avg

def spatialData_aggregation(decision_csv, velocity_csv, vorticity_csv, tke_csv, swirl_csv, strainRate_csv, avg_steps):

	"""
	This script aggreate information in to 5Hz.
	Input::
	velocity_csv: a csv file path that refers to 'vel_noheaders.csv', results from PTV analysis 
	vorticity_csv: a csv file path that refers to 'vorticity_noheaders.csv', results from PTV analysis
	tke_csv: a csv file path that refers to 'tke_noheaders.csv', results from PTV analysis
	swirl_csv: a csv file path that refers to 'swirl_noheaders.csv', results from PTV analysis, 05/24/2018
	strainRate_csv: a csv file path that refers to 'strainrate_noheaders.csv', results from PTV analysis , 05/24/2018

	Output::
	spatialData: a numpy ndarray in form of (nb_decisions, 6, 88, 157), 88 rows, 157 cols
	"""

	decisions_file = pd.read_csv(decision_csv)
	u_velocity, v_velocity = u_v(velocity_csv)
	vor = vor_tke(vorticity_csv)
	tke = vor_tke(tke_csv)
	swirl = vor_tke(swirl_csv)
	strainRate = vor_tke(strainRate_csv)

	# average the feature space over avg_steps consecutive frames, the return is in format of 
	# (avg_nb_decisions, nb_y_rows, nb_x_cols)
	# 05/25/2018, there are n datapts, in decision files, there are n-2 decisions. Acturally, there are n-1 decisions. Discard the 1st one because it has no history.
	avg_nb_decisions = len(decisions_file) + 1 

	### 06/01/2018
	u_velocity = average_feature(u_velocity, avg_nb_decisions, avg_steps)
	print "u_velocity shape:", np.shape(u_velocity)
	u_velocity_current_step = u_velocity[1:, :, :] # discard the 1st set of features
	u_velocity_1_step_behind = u_velocity[:avg_nb_decisions-1, :, :]
	
	u_acceleration = np.zeros((avg_nb_decisions-1, nb_y_rows, nb_x_cols))
	for i, (u1, u0) in enumerate(zip(u_velocity_current_step, u_velocity_1_step_behind)):
		u_acceleration[i, :, :] = u1 - u0

	assert np.shape(u_velocity_current_step) == np.shape(u_velocity_1_step_behind) == np.shape(u_acceleration)

	v_velocity = average_feature(v_velocity, avg_nb_decisions, avg_steps)
	v_velocity_current_step = v_velocity[1:, :, :] # discard the 1st set of features
	v_velocity_1_step_behind = v_velocity[:avg_nb_decisions-1, :, :]

	v_acceleration = np.zeros((avg_nb_decisions-1, nb_y_rows, nb_x_cols))
	for i, (v1, v0) in enumerate(zip(v_velocity_current_step, v_velocity_1_step_behind)):
		v_acceleration[i, :, :] = v1 - v0

	assert np.shape(v_velocity_current_step) == np.shape(v_velocity_1_step_behind) == np.shape(v_acceleration)

	vor = average_feature(vor, avg_nb_decisions, avg_steps)[1:, :, :]
	tke = average_feature(tke, avg_nb_decisions, avg_steps)[1:, :, :]
	swirl = average_feature(swirl, avg_nb_decisions, avg_steps)[1:, :, :]
	strainRate = average_feature(strainRate, avg_nb_decisions, avg_steps)[1:, :, :]
	print np.shape(u_velocity)
	print np.shape(v_velocity)
	print np.shape(vor)
	print np.shape(tke)
	print np.shape(swirl)
	print np.shape(strainRate)
	
	

	features_all = np.array([u_velocity[1:, :, :][:, ::-1, :], v_velocity[1:, :, :][:, ::-1, :], u_acceleration[:, ::-1, :], v_acceleration[:, ::-1, :], vor[:, ::-1, :], tke[:, ::-1, :], swirl[:, ::-1, :], strainRate[:, ::-1, :]])
	features_all = np.swapaxes(features_all, 0, 1)
	print np.shape(features_all)
	print len(decisions_file)
	assert (np.shape(features_all) == (len(decisions_file), 8, 88, 157)), "Number of grid points does not equal to number of velocity points"
	return features_all
 
def run(DIR, dist_threshold, avg_steps):
	
	# import meta data of the video clips
	global img_height, img_width, delta_t, nb_y_rows, nb_x_cols, ratio, delta_x, delta_y, nb_matrix
	meta_ = pd.read_csv(os.path.join(DIR, 'meta_data_videoclip.csv'))
	img_height = meta_['img_height'][0]
	img_width  = meta_['img_width'][0]
	delta_t    = meta_['delta_t'][0]
	nb_y_rows  = meta_['nb_y_rows'][0]
	nb_x_cols  = meta_['nb_x_cols'][0]
	ratio      = meta_['ratio'][0]

	delta_x = meta_['delta_x_y'][0]
	delta_y = meta_['delta_x_y'][0]

	nb_matrix = np.int(round(meta_['duration'][0]/meta_['delta_t'][0])) + 1

	# subdir that stores PTV data
	DIR_CLIP = DIR.split('/')[1].strip('\'')
	SUB_DIR = 'UndistortedPreprocessed/forPTV'
	PTV_CSVs = glob.glob(os.path.join(DIR, SUB_DIR, '*.csv'))
 	
	for file_path in PTV_CSVs:
		if file_path.split('/')[-1]  == 'vel_noheaders.csv':
			velocity_csv = file_path
		elif file_path.split('/')[-1]  == 'vorticity_noheaders.csv':
			vorticity_csv = file_path
		elif file_path.split('/')[-1]  == 'tke_noheaders.csv':
			tke_csv = file_path
		elif file_path.split('/')[-1]  == 'swirl_noheaders.csv':
			swirl_csv = file_path
		elif file_path.split('/')[-1]  == 'strainrate_noheaders.csv':
			strainRate_csv = file_path

	# decision file path
	DECISION_CSV = os.path.join(DIR, '{}_fishDecisions_{}_mm_5Hz.csv'.format(DIR_CLIP, dist_threshold))

	# # patch size, in grid points, 10 by 10 means 15cm by 15cm, which is approximate 3BL of emerald shiner.
	# patch_size = (10, 14)	

	print 'Saving extracted features (u, v, vor, tke, swirl, strainRate) and corresponding labels (decisions) into HDF5 format...'
	# Save preprocessed data as hdf5 format
	filename = os.path.join(DIR, 'environment_{}.h5'.format(DIR_CLIP))
	print filename
	envrionmentSet = spatialData_aggregation(DECISION_CSV, velocity_csv, vorticity_csv, tke_csv, swirl_csv, strainRate_csv, avg_steps)

	with h5py.File(filename, 'w') as w_hf:
		w_hf.create_dataset("envrionmentSet",  data=envrionmentSet)

	print 'Saving Completed!'
	print '----------------------------------------------------------------------' 

	# directory that store all csv files
	shutil.copy2(filename, batch_dir)

# Batch Processing, input the directory of all 53 video clips
meta_clips = pd.read_csv('clips_list_threshold.csv')
data_lists = meta_clips['video_clip']

dist_threshold = 10 # mm
avg_steps = 4 # average over 4 steps, to make 20Hz down to 5Hz

batch_dir = 'environmentAllTrjectories'
if not os.path.exists(batch_dir):
	os.mkdir(batch_dir)

for i, img_dir in enumerate(data_lists):

	# print i
	# print img_dir
	img_dir = img_dir.strip('\'')
	run(img_dir, dist_threshold, avg_steps)






'''
# only the trajectory "11_09_2017/FishBehavior_11_09_5" is reserved for virtual fish environment
data_list = '11_02_2017/FishBehavior_11_02_6'

dist_threshold = 10 # mm
avg_steps = 4 # average over 4 steps, to make 20Hz down to 5Hz


img_dir = data_list.strip('\'')
run(img_dir, dist_threshold, avg_steps)
'''
