import os

import sys

import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from plot import plot_history

import numpy as np
from numpy import save
from numpy import load
import cv2

import PySimpleGUI as GUI


model = None
model_classes_from_file = None


def TrainModel(dataset_directory_root, exec_directory_root, img_width, img_height):

	global model

	batch = 22

	input_shape = (img_width, img_height, 3)

	labels = os.listdir(dataset_directory_root) 


	Accuracy_Threshold = 0.9999
	loss_threshold = 0.02
	epochs = 0

	# MODEL 1

	# model = keras.Sequential([
	# 	layers.Conv2D(32, (3,3), input_shape=input_shape),
	# 	layers.Activation('relu'),
	# 	layers.MaxPooling2D(pool_size=(2,2)),

	# 	layers.Conv2D(32, (3,3)),
	# 	layers.Activation('relu'),
	# 	layers.MaxPooling2D(pool_size=(2,2)),

	# 	layers.Conv2D(64, (3, 3)),
	# 	layers.Activation('relu'),
	# 	layers.MaxPooling2D(pool_size=(2, 2)),

	# 	layers.Flatten(),
	# 	layers.Dense(64),
	# 	layers.Activation('relu'),
	# 	layers.Dropout(0.5),
	# 	layers.Dense(len(labels)),
	# 	layers.Activation('softmax'), #softmax / sigmoid
	# ])

	# MODEL 2

	# model = keras.Sequential()
	# model.add(layers.Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = input_shape))
	# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# model.add(keras.layers.Dropout(0.25))
	# model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(128, activation='relu'))
	# model.add(keras.layers.Dropout(0.5))
	# model.add(keras.layers.Dense(len(labels), activation='softmax'))

	# MODEL 3

	# img_input = layers.Input(shape=input_shape)
	# x = layers.Conv2D(32, 3, activation='relu')(img_input)
	# x = layers.Conv2D(64, 3, activation='relu')(x)
	# x = layers.Conv2D(128, 3, activation='relu')(x)
	# x = layers.MaxPooling2D(2, 2)(x)
	# x = layers.Dropout(0.25)(x)
	# x = layers.Flatten()(x)
	# x = layers.Dense(128, activation='relu')(x)
	# x = layers.Dropout(0.5)(x)
	# output = layers.Dense(len(labels), activation='softmax')(x)

	# model = Model(img_input, output)

	# MODEL 4

	# model = keras.Sequential([
 #        keras.Input(shape=input_shape),
 #        layers.Conv2D(8, kernel_size=(3, 3), kernel_regularizer=l1_l2(), activation="relu"),
 #        layers.MaxPooling2D(pool_size=(2, 2)),
 #        layers.Dropout(0.3),

 #        layers.Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l1_l2(), activation="relu"),
 #        layers.MaxPooling2D(pool_size=(2, 2)),
 #        layers.Dropout(0.3),

 #        layers.Flatten(),
 #        layers.Dropout(0.5),
 #        layers.Dense(len(labels), activation="softmax"),
 #    ])

 	# MODEL 5

	model = keras.Sequential()
	# add Convolutional layers
	model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
					input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))

	model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))

	model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))  

	model.add(layers.Flatten())
	model.add(layers.Dropout(0.2))
	# Densely connected layers
	model.add(layers.Dense(128, activation='relu'))
	# output layer
	model.add(layers.Dense(len(labels), activation='softmax'))




	datagen = ImageDataGenerator(
			rescale = 1. / 255,
			validation_split = 0.3,
			rotation_range=8,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.1,
			zoom_range=0.1,
		)

	ds_train = datagen.flow_from_directory(
			dataset_directory_root, 
			target_size = (img_width, img_height),
			batch_size = batch,
			class_mode = 'categorical',
			shuffle = True,
			seed = 123,
			subset = 'training',
		)

	ds_validate = datagen.flow_from_directory(
			dataset_directory_root, 
			target_size = (img_width, img_height),
			batch_size = batch,
			class_mode = 'categorical',
			shuffle = True,
			seed = 123,
			subset = 'validation',
		)

	class CallBack(tf.keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}):
			if((logs.get('accuracy') > Accuracy_Threshold) & (logs.get('loss') < loss_threshold) 
				& (logs.get('val_accuracy') > Accuracy_Threshold) & (logs.get('val_loss') < loss_threshold)):
				self.model.stop_training = True

	callbacks = CallBack()

	model.compile(
		loss='categorical_crossentropy',
		optimizer = keras.optimizers.Adam(learning_rate = 0.0005), #rmsprop, adam, SGD
		metrics = ['accuracy']
	)

#------------------------------------------------------

	GUI.theme('Light Blue 1')

	font = ('Arial', 13)

	training_prompt_text = [[GUI.Text('The training images have been generated.', font=font)],
							[GUI.Text('On the next screen you will be asked to type a number of epochs.', font=font)],
							[GUI.Text('When the training is complete, the results will appear\n'
									  'and you will be prompted to save the model or not.', font=font)],
							[GUI.Text('Please note that the training will continue even if you close the window.', font=('Arial',13))],
							[GUI.Text('If the program must be stopped during training, please kill it with the task manager.', font=font)]]
	
	training_prompt_continue = [[GUI.Button('I understand', key='continue', font=font, pad=(15,0)), GUI.Button('Exit', key='Exit', font=font, pad=(15,0))]]

	training_prompt = [[GUI.Col([
								 [GUI.Frame(layout = training_prompt_text, title = '', border_width = 0)],
								 [GUI.Frame(layout = training_prompt_continue, title = '', border_width = 0)]
								], element_justification = 'center')]]

	insert_epoch_text = [[GUI.Text('Please type the number of epochs that the model will be trained for.', font=font)],
						 [GUI.Text('The model is designed to stop its training once the Accuracy and Loss \n'
						 		   'values are better than a specified threshold.', font=font)],
						 [GUI.Text('If that threshold is not reached, then the training will stop once \n'
						 		   'the model reaches the maximum number of specified epochs.', font=font)]]

	insert_epoch_input = [[GUI.Input(key='input', enable_events=True), GUI.Button('OK', key='ok', font=('Arial',13))]]

	insert_epoch = [[GUI.Col([
							  [GUI.Frame(layout = insert_epoch_text, title = '', border_width = 0)],
							  [GUI.Frame(layout = insert_epoch_input, title = '', border_width = 0)]
							 ], element_justification = 'center')]]

	training_in_progress = [[GUI.Output(size=(80,20), key='out')],
							[GUI.Button('Finish', key='finish', font=font)]]

	layout = [[GUI.Column(training_prompt, visible=True, key='-Training_Prompt-'),
			   GUI.Column(insert_epoch, visible=False, key='-Insert_Epoch-'),
			   GUI.Column(training_in_progress, visible=False, key='-Training_In_Progress-')]]

	window = GUI.Window('Optical Character Recognition Demo', layout)

	while True:
		event, values = window.read()

		if event == GUI.WIN_CLOSED or event == 'Cancel':
			break

		if event == 'continue':
			window[f'-Training_Prompt-'].update(visible=False)
			window[f'-Insert_Epoch-'].update(visible=True)

		if event == 'input' and values['input'] and values['input'][-1] not in ('0123456789.'):
			window['input'].update(values['input'][:-1]) # Characters that are not digits will not appear in the input box.

		if event == 'ok':
			if values['input'] != '':
				epochs = int(values['input'])

				window[f'-Insert_Epoch-'].update(visible=False)
				window[f'-Training_In_Progress-'].update(visible=True)

				history = model.fit(ds_train, validation_data = ds_validate, epochs = epochs, verbose = 2, callbacks=[callbacks])
				print('\n\nTraining is Complete!')

		if ((event == 'finish') | (event == 'Exit')):
			break


	window.close()

#----------------------------------------------------
	try:
		plot_history(history)
	except:
		return

#--------------------------------------------------------------------

	save_model_prompt = [[GUI.Text('Type a name for your model', font=font)],
				  		 [GUI.Input(key='ModelName')],
				  		 [GUI.Button('Save Model', key='ModelSave', font=font, pad=(15,0)), GUI.Cancel(key='Cancel', font=font, pad=(15,0))]]

	model_saved_text = [[GUI.Text('Your model has been saved successfully.', font=font)],
				   		[GUI.Text('By pressing Finish, the program will restart\n'
				   			 	  'so you can try your created model.', font=font)]]
	
	model_saved_buttons = [[GUI.Button('Finish', key='model_save_finish', font=font, pad=(15,0)), GUI.Button('Exit', key='Exit', font=font, pad=(15,0))]]

	model_saved = [[GUI.Col([
							  [GUI.Frame(layout = model_saved_text, title = '', border_width = 0)],
							  [GUI.Frame(layout = model_saved_buttons, title = '', border_width = 0)]
							 ], element_justification = 'center')]]

	layout = [[GUI.Column(save_model_prompt, visible = True, key='-Save_Model-'),
			   GUI.Column(model_saved, visible = False, key='-Model_Saved-')]]

	window = GUI.Window('Optical Character Recognition Demo', layout)

	model_name = ''

	while True:
		event, values = window.read()
		
		if event == GUI.WIN_CLOSED:
			break

		if event == 'ModelSave':
			model_name = values['ModelName']
			if (model_name != ''):

				if (re.search('[\\/:"*?<>|]+', model_name)):

					GUI.popup('The name is invalid.', font=font)
				else:

					final_model_name = model_name + '.h5'
					classes_name = model_name + '.npy'

					os.chdir(exec_directory_root + '/Models')
					model.save(final_model_name) #Save a trained model

					classes = ds_train.class_indices
					model_classes = np.array(classes)
					save(classes_name, model_classes)

					window[f'-Save_Model-'].update(visible=False)
					window[f'-Model_Saved-'].update(visible=True)

			else:
				GUI.popup('The name may not be empty.', font=font)

		if ((event == 'Cancel') | (event == 'Exit')):
			break

		if event == 'model_save_finish':
			os.execv(sys.executable, ['python'] + sys.argv)

	window.close()


#--------------------------------------------------------------------


def Predict(images):
	global model
	global model_classes_from_file

	output_label = ''

	classes = np.load(model_classes_from_file, allow_pickle=True)
	classes_list = classes.tolist()
	classes_names = []
	for c in classes_list:
		classes_names.append(c)

	processed_images = ImagePreProcessing(images)

	for pi in processed_images:

		prediction = model.predict(pi)
		classification = prediction.argmax(axis = -1)
		output_label = output_label + classes_names[int(classification)]

	# print(output_label)
	return output_label


def LoadModel(model_path, classes_path):
	global model
	global model_classes_from_file

	model = load_model(model_path)	#OCR_MODEL_1.h5
	model_classes_from_file = classes_path

	# MakePrediction()



def ImagePreProcessing(images):

	processed_images = []

	for i in images:
		gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)	# Converts the image to grayscale

		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	# Inverts the image

		RGB_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		
		img_arr = img_to_array(RGB_thresh)
		img_arr = img_arr / 255
		img_arr = np.expand_dims(img_arr, axis = 0)

		# cv2.imshow('Processed Image', RGB_thresh)	#Debug - Display processed image
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# print(img_arr.shape)
		# img_arr = img_arr.reshape(-1, 50, 50, 3)

		processed_images.append(img_arr)

	return processed_images