import json
import h5py
import os
import random
import numpy as np 
import pandas as pd 
import keras.backend as K 

from stepLength_motionAngle import densityEstimation, trainKDEs

from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from keras.models import Model, Sequential, model_from_json
from keras.utils.np_utils import to_categorical


### Helper Functions
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


def under_sampling(X, s, a, y):
	"""
	under sample the input data 
	"""
	XShape = np.shape(X)
	X = X.reshape(XShape[0], -1) # reshape X into 2d array
	s = s.reshape(len(s), -1)
	a = a.reshape(len(a), -1)
	ratio = 'auto'
	X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=789).fit_sample(X, y)
	s_res, y_res = RandomUnderSampler(ratio=ratio, random_state=789).fit_sample(s, y)
	a_res, y_res = RandomUnderSampler(ratio=ratio, random_state=789).fit_sample(a, y)
	newLen = (np.shape(X_res)[0],)
	X_res = X_res.reshape(newLen+XShape[1:])
	return X_res, s_res, a_res, y_res


def scale_transform(scaler_X, X):
	"""
	transform the input data with scaler trained with training data
	"""

	X_shape = np.shape(X)
	X = X.flatten()[:, np.newaxis]
	X_scaled = scaler_X.transform(X).reshape(X_shape)
	return X_scaled


def sensoryInputScaler(trainData):

	"""
	read data, split it into train/test
	"""
	# step 1: read training data
	with h5py.File(trainData, 'r') as dataset:
		X = dataset['featureSetRaw'].value
		s = dataset['fishSpeeds'].value     
		a = dataset['fishDecisionAngles'].value
		y = dataset['decisionLabels'].value

	# step 2: undersample the training data
	X, s, a, y = under_sampling(X, s, a, y)
	
	# step 3: randomly shuffle the training data
	random_seed = 6789
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = random_seed)
	s_train, s_test, y_train, y_test = train_test_split(s, y, test_size = 0.2, stratify=y, random_state = random_seed)
	a_train, a_test, y_train, y_test = train_test_split(a, y, test_size = 0.2, stratify=y, random_state = random_seed)

	# step 4: define scaler
	scaler = QuantileTransformer(output_distribution='normal')

	# step 5: fit scaler with training data
	u_v = (np.concatenate((X_train[:, 0, :, :], X_train[:, 1, :, :]), axis = 0).flatten()[:,np.newaxis])
	u_v_scaler = scaler.fit(u_v)
	# u_v_a_scaler = scaler.fit(np.concatenate((X_train[:, 2, :, :], X_train[:, 3, :, :]), axis = 0).flatten()[:,np.newaxis])
	vor_scaler = scaler.fit(X_train[:, 2, :, :].flatten()[:, np.newaxis])
	tke_scaler = scaler.fit(X_train[:, 3, :, :].flatten()[:, np.newaxis])
	swirl_scaler = scaler.fit(X_train[:, 4, :, :].flatten()[:, np.newaxis])
	strainRate_scaler = scaler.fit(X_train[:, 5, :, :].flatten()[:, np.newaxis])
	speed_scaler = scaler.fit(s_train)
	decisionAngle_scaler = scaler.fit(a_train)


	return u_v_scaler, vor_scaler, tke_scaler, swirl_scaler, strainRate_scaler, speed_scaler, decisionAngle_scaler


def virtualFishModel(modelStructure, modelWeights):
	"""
	Input::
	modelStructure:  the structure of trained model 
	modelWeights: the weights for the trained model

	Ouput::
	loaded_model: the trained model with loaded weights
	"""


	# load json and create model
	json_file = open(modelStructure, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load model_weights
	loaded_model.load_weights(modelWeights)
	return loaded_model


def Reconstruct_trajectory(motionAngles, stepLengths, p0, decision, random_seed):
	"""
	Input::

	p0: fish current position (x, y) in mm.
	decision: action that fish chose at current position
	motionAngles: a dictionary containing all esitmated density distribution functions of motion angles
	stepLengths: a dictionary containing all esitmated density distribution functions of step lengths


	Ouput::

	nextLocation: a numpy array containing fish next position
	fishMotionAngle: the motion angle sample drawn from distribution, which is used as "privious motion angle" for next fish decision prediction.
	fishSpeed: the fish speed drawn from distribution, which is used as "previous fish speed" for next fish decision prediction
	"""
	x_0, y_0 = p0[0], p0[1]
	x_new, y_new = 0, 0

	if decision == 4: # this is a rare fish decision, which is not included in virtual fish training.
		x_new = x_0 + random.uniform(0, 10) 
		y_new = y_0 + random.uniform(0, 10)

		nextLocation = [x_new, y_new] # np.array([x_new, y_new])
		fishMotionAngle = 0
		fishSpeed = 0

	else:
		fishMotionAngle = motionAngles[str(decision)].sample(1, random_state = random_seed)
		fishStepLength = stepLengths[str(decision)].sample(1, random_state = random_seed)
		fishSpeed = fishStepLength * 0.001 * 5 # convert to m/s
		# print fishMotionAngle
		# print fishStepLength
		# print "~~~~~~~~~~~~~~~"

		# fish new location
		x_new = x_0 + np.cos(fishMotionAngle * np.pi) * fishStepLength
		y_new = y_0 + np.sin(fishMotionAngle * np.pi) * fishStepLength
		nextLocation = [x_new, y_new] # np.array([x_new, y_new])

	return nextLocation, fishMotionAngle, fishSpeed


### Fish Action Principles
def virtualFish_Action(virtualFish, feature_u, feature_v, feature_vor, feature_tke, feature_swirl, feature_strainRate, previousFishSpeed, previousMotionAngle):
	"""
	Input::
	virtualFish: loaded train model
	sensoryInputs: (u, v, vorticity, tke, swirl, strain rate), in form of (6, 10, 14)
	previousFishSpeed: previous fish action speed, which is randomly drawn from an estimated density funtion.
	previousMotionAngle: previous fish action angle, which is randomly drawn from an estimated density funtion.

	Output::
	action: an integer, which indicates fish action. 0: holding position, 1: forward, 2: left turn, 3: right turn
	"""

	# scale sensory inputs
	u_feature = scale_transform(u_v_scaler, feature_u)
	v_feature = scale_transform(u_v_scaler, feature_v)
	vor_feature = scale_transform(vor_scaler, feature_vor).reshape(1, 1, 10, 14)
	tke_feature = scale_transform(tke_scaler, feature_tke).reshape(1, 1, 10, 14)
	swirl_feature = scale_transform(swirl_scaler, feature_swirl).reshape(1, 1, 10, 14)
	strainRate_feature = scale_transform(strainRate_scaler, feature_strainRate).reshape(1, 1, 10, 14)

	# scale initial action and speed
	print previousFishSpeed
	print np.shape(previousFishSpeed)
	previousSpeed_feature = scale_transform(speed_scaler, previousFishSpeed)
	previousMotionAngle_feature = scale_transform(decisionAngle_scaler, previousMotionAngle)

	feature_u_v = np.asarray([u_feature, v_feature]).reshape(1, 2, 10, 14) #np.concatenate((u_feature, v_feature), axis=1)
	feature_speed_angle = np.concatenate((previousSpeed_feature, previousMotionAngle_feature), axis=1)

	scaled_Inputs = [feature_u_v, vor_feature, tke_feature, swirl_feature, strainRate_feature, feature_speed_angle]

	virtualFish_pred_prop = virtualFish.predict(scaled_Inputs) 
	action = np.argmax(virtualFish_pred_prop, axis=1)[0]
	return action, virtualFish_pred_prop

### Virtual Fish Run Functions
def run(k, x_0, y_0, action_0, virtualFishModelStructure, virtualFishModelWeights, data_path_allFishDecisions, fish_testEnvironment, random_seed):
	
	# obtain esimated density functions for fish motion angles and step lengths
	kde_fishDecisionAngles_0, kde_fishDecisionAngles_1, kde_fishDecisionAngles_2, kde_fishDecisionAngles_3, kde_currentStepLength_0, kde_currentStepLength_1, kde_currentStepLength_2, kde_currentStepLength_3 = trainKDEs(data_path_allFishDecisions)

	# build a dictionary to call corresponding esitmated density distribution function
	motionAngles = {"0":kde_fishDecisionAngles_0, "1": kde_fishDecisionAngles_1, "2": kde_fishDecisionAngles_2, "3": kde_fishDecisionAngles_3}
	stepLengths = {"0":kde_currentStepLength_0, "1": kde_currentStepLength_1, "2": kde_currentStepLength_2, "3": kde_currentStepLength_3}
	
	# load virtual fish model
	virtualFish = virtualFishModel(virtualFishModelStructure, virtualFishModelWeights)

	# import environment sequence
	with h5py.File(fish_testEnvironment, 'r') as dataset:
		"""
		an environment sequence, in format of (n, 6, 88, 157), which has n time steps, each one has 6 attributes in form of 88 rows and 157 cols.
		"""
		environmentSequence = dataset['envrionmentSet'].value
	
	# step 1: 'initialSpeed' and 'initialMotionAngle' take fish to next position, in mm
	new_location, fishSpeed, fishMotionAngle = Reconstruct_trajectory(motionAngles, stepLengths, [x_0, y_0], action_0, random_seed)
	new_location_x, new_location_y = new_location[0], new_location[1]

	Predicted_actions = [action_0]
	Pts_reconstruct = [[x_0, y_0]]

	# start from the 2nd position.
	for i in range(1, len(environmentSequence)):
		print i
		# step 2: use fish position to locate its sensory inputs, and scale them
			# - convert mm into grid coordinates, and translate them by patch_width/2 and patch_height/2, to compensate zero-padding. 
		x_c_grid = np.int(round(new_location_x/delta_x)) + patch_width/2
		y_c_grid = np.int(round(new_location_y/delta_y)) + patch_height/2
			# - use x_c_grid, y_c_grid to compute the sensory area. Use this to slice velocity, tke and vorticity vectors
			# - left, right, up and down are in picture-coordinates.
		left  = np.int(round(x_c_grid-0.5*patch_width))
		up    = np.int(round(y_c_grid-0.5*patch_height))
		right = np.int(round(x_c_grid+0.5*patch_width))
		down  = np.int(round(y_c_grid+0.5*patch_height))
			# - locate instant u, v, vor, tke, swirl and strain rate with idx, patch, size to slice in picture-cooridnates
		u_instant = zeros_pad(environmentSequence[i, 0, :, :], patch_width, patch_height)
		v_instant = zeros_pad(environmentSequence[i, 1, :, :], patch_width, patch_height)
		vor_instant = zeros_pad(environmentSequence[i, 2, :, :], patch_width, patch_height)
		tke_instant = zeros_pad(environmentSequence[i, 3, :, :], patch_width, patch_height)
		swirl_instant = zeros_pad(environmentSequence[i, 4, :, :], patch_width, patch_height)
		strainRate_instant = zeros_pad(environmentSequence[i, 5, :, :], patch_width, patch_height)
			# - slice sensory inputs from current instant
		feature_u = np.nan_to_num(u_instant[up:down, left:right][::-1])
		feature_v = np.nan_to_num(v_instant[up:down, left:right][::-1])
		feature_vor = np.nan_to_num(vor_instant[up:down, left:right][::-1])
		feature_tke = np.nan_to_num(tke_instant[up:down, left:right][::-1])
		feature_swirl = np.nan_to_num(swirl_instant[up:down, left:right][::-1])
		feature_strainRate = np.nan_to_num(strainRate_instant[up:down, left:right][::-1])

		# step 3: use sensory inputs and the initial 'currentSpeed', and 'currentMotionAngle' to predict action
		new_action, new_action_prop = virtualFish_Action(virtualFish, feature_u, feature_v, feature_vor, feature_tke, feature_swirl, feature_strainRate, fishSpeed, fishMotionAngle)
		Predicted_actions.append(new_action)
		# if np.max(new_action_prop)<=0.4:
		# 	new_action = 1
		# else:
		# 	new_action = new_action
		# step 4: according to the action, calculate next location
		nextLocation, currentFishMotionAngle, currentFishSpeed = Reconstruct_trajectory(motionAngles, stepLengths, [new_location_x, new_location_y], new_action, random_seed)

		# step 5: reassign next_location, fishMotionAngle, fishSpeed as inputs
		nextLocation = np.asarray(nextLocation).ravel()

		new_location_x = nextLocation[0]
		new_location_y = nextLocation[1]

		fishSpeed = currentFishSpeed
		fishMotionAngle = currentFishMotionAngle
		print new_location_x, new_location_y
		Pts_reconstruct.append([new_location_x, new_location_y])

	# wirte the list locations into txt
	np.savetxt(os.path.join(DIR_VIRTUALFISH_TRIAL, 'VirtualFishTrajectory_{}.txt'.format(k)), np.asarray(Pts_reconstruct))
	np.savetxt(os.path.join(DIR_VIRTUALFISH_TRIAL, 'VirtualFishActions_{}.txt'.format(k)), np.asarray(Predicted_actions))
	print "Done!"

### run virtual fish in virtual environment ###
# global variables
delta_x = delta_y = 15
patch_width = 14
patch_height = 10
DATA_DIR = '/PATH TO PROJECT FILES/'
trainData = os.path.join(DATA_DIR,'PreprocessedData_10mm_patch_10h_14w_raw_6features/preprocessedData_patch_10h_14w_6features_raw.h5')
virtualFishModelStructure = os.path.join(DATA_DIR,'PreprocessedData_10mm_patch_10h_14w_raw_6features/xcepModelWithSpeed_ModelResults/0.6037_0.5792/incepModel_withSpeed_Dilation.json')
virtualFishModelWeights = os.path.join(DATA_DIR,'PreprocessedData_10mm_patch_10h_14w_raw_6features/xcepModelWithSpeed_ModelResults/0.6037_0.5792/incepModel_withSpeed_Dilation_weights.h5')
data_path_allFishDecisions = os.path.join(DATA_DIR,'fishDecisionFiles_10mm_5Hz/')
u_v_scaler, vor_scaler, tke_scaler, swirl_scaler, strainRate_scaler, speed_scaler, decisionAngle_scaler = sensoryInputScaler(trainData)

# run one trial multiple times
# Batch Processing, input the directory of all 52 video clips
meta_clips = pd.read_csv('/Research/Fish Passage Project/Fish Passage Experiments Data/clips_list_threshold.csv')
data_lists = meta_clips['video_clip']

for img_dir in data_lists:
	trial_name = img_dir.split('/')[1][13:]
	print trial_name

	# make a new directory to store all trials of virtual fish trajectories
	DIR_VIRTUALFISH = 'VirtualFishTrajectories'

	DIR_VIRTUALFISH_TRIAL = os.path.join(DIR_VIRTUALFISH, trial_name)
	if not os.path.exists(DIR_VIRTUALFISH_TRIAL):
		os.mkdir(DIR_VIRTUALFISH_TRIAL)

	decisionFile = pd.read_csv(os.path.join(data_path_allFishDecisions, 'FishBehavior_{}_fishDecisions_10_mm_5Hz.csv'.format(trial_name)))
	print len(decisionFile)
	x_0 = decisionFile['Current_location_cx_mm'][0]
	y_0 = decisionFile['Current_location_cy_mm'][0]
	action_0 = decisionFile['currentDecisions'][0]
	fish_testEnvironment = 'environmentAllTrjectories/environment_FishBehavior_{}.h5'.format(trial_name)

	random_seeds = [888, 6850, 6521, 9491, 85244,2415267]
	for k, rs in enumerate(random_seeds):
		run(k, x_0, y_0, action_0, virtualFishModelStructure, virtualFishModelWeights, data_path_allFishDecisions, fish_testEnvironment, rs)

