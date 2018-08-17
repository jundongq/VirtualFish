"""
This script is for plotting the average of virtual fish trajectories in physical modeling.
"""
import os
import glob
import h5py
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt 

from collections import defaultdict

### Define processing functions
def convert_to_dict(txt_file):
	"""
	This function is to convert the fish coordinates in one trajectory into a dictionary, with x coordinate as key, y coordinate as value
	
	Input::

	txt_file: a string, file handle referring to one virtual fish trajectory

	Ouput::
	coordinates_dict: a dictionary with x coordinate as key, y coordinate as value. It allows one key to have multiple values

	"""
	xy = np.loadtxt(txt_file)
	virtual_x_ccordinate = xy[1:,0]
	virtual_y_ccordinate = xy[1:,1]
	convert_x = lambda x: np.int(round(x,0)) # define a function to convert x coordinates to int
	vfunc = np.vectorize(convert_x)
	keys = vfunc(virtual_x_ccordinate)
	data_zip = zip(keys, virtual_y_ccordinate)

	coordinates_dict = defaultdict(list)
	for key, value in data_zip:
		coordinates_dict[key].append(value)

	return coordinates_dict


def avg_dict(coordinates_dict):
	"""
	This function is to average the y coordinates over x
	"""
	avg_d = {}
	for key, value in coordinates_dict.iteritems():
		avg_d[key] = sum(value)/float(len(value))
	return avg_d


def merge_dicts(list_dicts):
	"""
	This function is to merge a list of dictionaries. Each dictionary contains information for one fish trajectory
	"""

	merged_dict = defaultdict(list)

	for d in list_dicts:
		for key, value in d.iteritems():
			merged_dict[key].append(value)
	return merged_dict


def avg_routes_over_x(avg_routes, step):
	"""
	This function is to averge the fish coordinates over x, smooth the trajectory trend

	Input::
	avg_routes: the averaged routes over all trajectories
	step: number of steps averaged over x coordinates

	Ouput::
	avg_keys: averaged x coordinates for average of input(observed) fish trajectories
	avg_values: averaged y coordinates for average of input(observed) fish trajectories
	"""

	n = int(np.ceil(len(avg_routes)/float(step)))

	# step 1: averge the values of avg_routes, which indicates y coordinate
	avg_values = []
	y_coordinates_std = []
	for i in range(n):
		avg_values.append(np.mean(avg_routes.values()[i*step:(i+1)*step]))
		y_coordinates_std.append(np.std(avg_routes.values()[i*step:(i+1)*step]))

	# step 2: averge the keys of avg_routes, which indicates x coordinate
	avg_keys = []
	for i in range(n):
		avg_keys.append(np.mean(avg_routes.keys()[i*step:(i+1)*step]))
	return avg_keys, avg_values, y_coordinates_std


def run(DATA_DIR, step):
	"""
	Input::
	DATA_DIR: the directory containing all fish trajectory files
	step: number of steps averaged over x coordinates, to smooth the averaged trajectory trend

	"""
	# step 0: grab a list of file handles referring to the list of observed fish trajectories
	virtualFishData = glob.glob(os.path.join(DATA_DIR, '*/VirtualFishTrajectory_*.txt'))
	print len(virtualFishData)
	# step 1: convert dataframe[] and dataframe[] to dictionaries, with x as key, y as values
	all_coordinates_dicts = []
	for i in range(len(virtualFishData)):
		coordinates_dict = convert_to_dict(virtualFishData[i])

		# step 2: averge y values with same key x
		coordinates_dict_avg = avg_dict(coordinates_dict)
		all_coordinates_dicts.append(coordinates_dict_avg)

	# step 3: merge all dictionaries together
	all_routes = merge_dicts(all_coordinates_dicts)

	# step 4: average y values again
	avg_routes = avg_dict(all_routes)

	# step 5: average the avg_routes over x and y to reduce number of values
	x_avg, y_avg, y_coordinates_std = avg_routes_over_x(avg_routes, 10)

	# wirte the list locations into txt
	np.savetxt('averageofVirtualFishTrajectory.txt', np.asarray([x_avg, y_avg, y_coordinates_std]))
	print 'Done!'

### Run
## virtual fish trajectories
DATA_DIR = 'VirtualFishTrajectories/'
step = 10
run(DATA_DIR, step)




