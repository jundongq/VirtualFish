"""
Build a deep learning model to learn and predict fish decisions based on its surrounding features (u, v, vor, tke, swirl, strain rate).
Add speed as input 
"""


import h5py
import time
import json
import numpy as np 
import matplotlib.pylab as plt

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing.data import QuantileTransformer

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.wrappers.scikit_learn import KerasClassifier

import keras.backend as K 
import keras
from keras.models import Model, Sequential, model_from_json
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras import optimizers
from keras import regularizers

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model

# Import dataset
# the .h5 file contains preprocessed (scaled) dataset (dataset_size, channel, height, width)
preprocessedData = 'preprocessedData_patch_10h_14w_6features_raw.h5'

def under_sampling(X, s, a, y):

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


def read_split(data):
	"""
	read data, split it into train/test
	"""
	with h5py.File(data, 'r') as dataset:
		X = dataset['featureSetRaw'].value
		s = dataset['fishSpeeds'].value     
		a = dataset['fishDecisionAngles'].value
		y = dataset['decisionLabels'].value

	selector = np.asarray([True for i in range(len(y))])
	selector[1477:1483] = False # the elements in this range are noisy, the corresponding are the first 6 frames in "11_09_5".


	X = X[selector, :, :, :]
	s = s[selector]
	a = a[selector]
	y = y[selector]

	# selector2, remove lable 4
	selector2 = np.asarray([False if i == 4 else True for i in y])
	X = X[selector2, :, :, :]
	s = s[selector2]
	a = a[selector2]
	y = y[selector2]

	# undersample the training data
	X, s, a, y = under_sampling(X, s, a, y)
	
	# return X, s, a, y
	# randomly shuffle the data
	random_seed = 6789
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = random_seed)
	s_train, s_test, y_train, y_test = train_test_split(s, y, test_size = 0.2, stratify=y, random_state = random_seed)
	a_train, a_test, y_train, y_test = train_test_split(a, y, test_size = 0.2, stratify=y, random_state = random_seed)

	y_train = to_categorical(y_train)

	return X_train, X_test, s_train, s_test, a_train, a_test, y_train, y_test

# parameter settings
nb_classes = 4

input_shape_1 = (2, 10, 14) # u, v
input_shape_2 = (1, 10, 14) # vorticity
input_shape_3 = (1, 10, 14) # tke
input_shape_4 = (1, 10, 14) # swirl
input_shape_5 = (1, 10, 14) # strain rate
input_shape_6 = (2,) # speed, fish decision angle


def incepModel(L1_lambda, L2_lambda):

	### input
	input_data_1 = Input(shape = input_shape_1)
	input_data_2 = Input(shape = input_shape_2)
	input_data_3 = Input(shape = input_shape_3)
	input_data_4 = Input(shape = input_shape_4)
	input_data_5 = Input(shape = input_shape_5)
	input_data_6 = Input(shape = input_shape_6)


	### tower 1, u, v
	tower_1 = Conv2D(filters = 16,
					kernel_size = (3, 3),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					input_shape = input_shape_1,
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_1_1')(input_data_1)
	tower_1 = BatchNormalization(axis=1, name = 'BN_1_1')(tower_1)
	tower_1 = Conv2D(filters = 32,
					kernel_size = (1, 1),
					strides=(1, 1),
					padding = 'same',
					data_format = 'channels_first',
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_1_2')(tower_1)
	tower_1 = BatchNormalization(axis=1, name = 'BN_1_2')(tower_1)

	tower_1 = Conv2D(filters = 48,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_1_3')(tower_1)
	tower_1 = BatchNormalization(axis=1, name = 'BN_1_3')(tower_1)

	tower_1 = Conv2D(filters = 64,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_1_4')(tower_1)
	tower_1 = BatchNormalization(axis=1, name = 'BN_1_4')(tower_1)

	### tower 2, vorticity
	tower_2 = Conv2D(filters = 16,
					kernel_size = (3, 3),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					input_shape = input_shape_2,
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_2_1')(input_data_2)
	tower_2 = BatchNormalization(axis=1, name = 'BN_2_1')(tower_2)


	tower_2 = Conv2D(filters = 48,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_2_3')(tower_2)
	tower_2 = BatchNormalization(axis=1, name = 'BN_2_3')(tower_2)

	tower_2 = Conv2D(filters = 64,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_2_4')(tower_2)
	tower_2 = BatchNormalization(axis=1, name = 'BN_2_4')(tower_2)
	
	### tower 3, tke
	tower_3 = Conv2D(filters = 16,
					kernel_size = (3, 3),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					input_shape = input_shape_3,
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_3_1')(input_data_3)
	tower_3 = BatchNormalization(axis=1, name = 'BN_3_1')(tower_3)

	tower_3 = Conv2D(filters = 48,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_3_3')(tower_3)
	tower_3 = BatchNormalization(axis=1, name = 'BN_3_3')(tower_3)

	tower_3 = Conv2D(filters = 64,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_3_4')(tower_3)
	tower_3 = BatchNormalization(axis=1, name = 'BN_3_4')(tower_3)

	### tower 4, swirl
	tower_4 = Conv2D(filters = 16,
					kernel_size = (3, 3),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					input_shape = input_shape_4,
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_4_1')(input_data_4)
	tower_4 = BatchNormalization(axis=1, name = 'BN_4_1')(tower_4)

	tower_4 = Conv2D(filters = 48,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_4_3')(tower_4)
	tower_4 = BatchNormalization(axis=1, name = 'BN_4_3')(tower_4)

	tower_4 = Conv2D(filters = 64,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_4_4')(tower_4)
	tower_4 = BatchNormalization(axis=1, name = 'BN_4_4')(tower_4)

	### tower 5, strain rate
	tower_5 = Conv2D(filters = 16,
					kernel_size = (3, 3),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					input_shape = input_shape_5,
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_5_1')(input_data_5)
	tower_5 = BatchNormalization(axis=1, name = 'BN_5_1')(tower_5)

	tower_5 = Conv2D(filters = 48,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_5_3')(tower_5)
	tower_5 = BatchNormalization(axis=1, name = 'BN_5_3')(tower_5)

	tower_5 = Conv2D(filters = 64,
					kernel_size = (2,2),
					strides=(1, 1),
					padding = 'valid',
					data_format = 'channels_first',
					dilation_rate = (2, 2),
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_5_4')(tower_5)
	tower_5 = BatchNormalization(axis=1, name = 'BN_5_4')(tower_5)

	### tower 6, only 1x1 dense layer

	tower_6 = Dense(48,
					input_shape = input_shape_6,
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_6_1')(input_data_6)
	

	output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis=1)

	output = Conv2D(filters = 320,
					kernel_size = (1, 1),
					strides=(1, 1),
					padding = 'same',
					data_format = 'channels_first',
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_all_1')(output)
	output = GlobalAveragePooling2D(data_format = 'channels_first', name='GAP_1')(output)
	
	output = keras.layers.concatenate([output, tower_6], axis=1)

	output = Dense(128,
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_output_1')(output)
	output = Dense(48,
					activation = 'relu',
					kernel_initializer = 'glorot_normal',
					kernel_regularizer = regularizers.l1_l2(L1_lambda, L2_lambda),
					name = 'tower_output_2')(output)

	out    = Dense(nb_classes, activation = 'softmax', kernel_initializer = 'glorot_normal', name='Output')(output)
	model  = Model(inputs = [input_data_1,input_data_2,input_data_3, input_data_4, input_data_5, input_data_6], outputs = out)

	return model

model = incepModel(0.002, 0.0001)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=K.epsilon(), decay=0.0)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
print model.summary()
plot_model(model, to_file='xceptionModel_withSpeed_Dilation_plot.png', show_shapes=True, show_layer_names=True)

# save layer names into a set, to visualize all layers' output in tensorboard
embeddings_all_layer_names = set(layer.name for layer in model.layers if layer.name.startswith('tower_'))
print embeddings_all_layer_names


# save the model into json
modelToJson = model.to_json()
with open ('incepModel_withSpeed_Dilation.json', 'w') as wf:
	wf.write(modelToJson)

X_train, X_test, s_train, s_test, a_train, a_test, y_train, y_test = read_split(preprocessedData)

# Standarize X_train, s_train

# step 1: define scaler
scaler = QuantileTransformer(output_distribution='normal')

u_v_scaler = scaler.fit(np.concatenate((X_train[:, 0, :, :], X_train[:, 1, :, :]), axis = 0).flatten()[:,np.newaxis])
vor_scaler = scaler.fit(X_train[:, 2, :, :].flatten()[:, np.newaxis])
tke_scaler = scaler.fit(X_train[:, 3, :, :].flatten()[:, np.newaxis])
swirl_scaler = scaler.fit(X_train[:, 4, :, :].flatten()[:, np.newaxis])
strainRate_scaler = scaler.fit(X_train[:, 5, :, :].flatten()[:, np.newaxis])
speed_scaler = scaler.fit(s_train)
decisionAngle_scaler = scaler.fit(a_train)

def scale_transform(scaler_X, X):
	X_shape = np.shape(X)
	X = X.flatten()[:, np.newaxis]
	X_scaled = scaler_X.transform(X).reshape(X_shape)
	return X_scaled


input_1 = scale_transform(u_v_scaler, X_train[:,:2,:,:])
input_2 = scale_transform(vor_scaler, X_train[:,[2],:,:])
input_3 = scale_transform(tke_scaler, X_train[:,[3],:,:])
input_4 = scale_transform(swirl_scaler, X_train[:,[4],:,:])
input_5 = scale_transform(strainRate_scaler, X_train[:,[5],:,:])

input_6_1 = scale_transform(speed_scaler, s_train)
input_6_2 = scale_transform(decisionAngle_scaler, a_train)
# concatenate speed and decision angle together
input_6 = np.concatenate((input_6_1, input_6_2), axis=1)


# train and save the model weights
xcepModelVariant_weights_path = 'incepModel_withSpeed_Dilation_weights.h5'

t0 = time.time()
checkpointer = ModelCheckpoint(xcepModelVariant_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

earlystopping = EarlyStopping(monitor='val_loss', patience=10)

tensorboard_log_dir = 'xcepModelWithSpeedVariantLogs/{}'.format(time.time())
tensorboard  = TensorBoard(log_dir = tensorboard_log_dir, histogram_freq = 1, 
				write_graph=True, write_images=True, embeddings_freq=1, 
				embeddings_layer_names=embeddings_all_layer_names, embeddings_metadata=None)

callbacks_list = [checkpointer, reduce_lr, earlystopping, tensorboard]

history = model.fit([input_1,input_2, input_3, input_4, input_5, input_6], y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=callbacks_list)
t1 = time.time()
t = t1-t0
print 'The incepModel took %.2f mins.' %(round(t/60., 2))

# load json and create model
json_file = open('incepModel_withSpeed_Dilation.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load model_weights
loaded_model.load_weights(xcepModelVariant_weights_path)

# check out model performance on test dataset.
test_input_1 = scale_transform(u_v_scaler, X_test[:,:2,:,:])
test_input_2 = scale_transform(vor_scaler, X_test[:,[2],:,:])
test_input_3 = scale_transform(tke_scaler, X_test[:,[3],:,:])
test_input_4 = scale_transform(swirl_scaler, X_test[:,[4],:,:])
test_input_5 = scale_transform(strainRate_scaler, X_test[:,[5],:,:])

test_input_6_1 = scale_transform(speed_scaler, s_test)
test_input_6_2 = scale_transform(decisionAngle_scaler, a_test)
# concatenate speed and decision angle together
test_input_6 = np.concatenate((test_input_6_1, test_input_6_2), axis=1)

pred_prop = loaded_model.predict([test_input_1, test_input_2, test_input_3, test_input_4, test_input_5, test_input_6]) # it returns probability for each class
y_pred = np.argmax(pred_prop, axis=1)
print "The first 20 ground truth labels:"
print y_test[:20]
print "The first 20 predicted labels:"
print y_pred[:20]
print "The acc score is: ", accuracy_score(y_test, y_pred)
print "The F1 score is: ", f1_score(y_test, y_pred, average='weighted')
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# confusion matrix
labels = ['Holding Position(0)', 'Forward(1)', 'Left Turn(2)', 'Right Turn(3)'] #, 'Backward(4)']
print classification_report(y_test, y_pred, target_names=labels)
print confusion_matrix(y_test, y_pred)


