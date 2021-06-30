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

from plot import plot_training_results

import numpy as np
from numpy import save
from numpy import load
import cv2

import PySimpleGUI as GUI


model = None
model_classes_from_file = None


def TrainModel(dataset_directory_root, exec_directory_root, img_width, img_height):

	global model

	batch = 22 #Images batch size.

	input_shape = (img_width, img_height, 3) #Default (50, 50, 3)

	labels = os.listdir(dataset_directory_root) #Label names of the images.
												#Taken from the folder names.

	Accuracy_Threshold = 0.9999 #Given threshold for accuracy
	loss_threshold = 0.02 #Given threshold for loss

	epochs = 0 #Epochs initialization

	# =============== MODEL 1 ===========================

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

	# ================== MODEL 2 ===========================

	# model = keras.Sequential()
	# model.add(layers.Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = input_shape))
	# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# model.add(keras.layers.Dropout(0.25))
	# model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(128, activation='relu'))
	# model.add(keras.layers.Dropout(0.5))
	# model.add(keras.layers.Dense(len(labels), activation='softmax'))

	# =============== MODEL 3 =============================

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

	# ================ MODEL 4 =============================

	# model = keras.Sequential([
	# keras.Input(shape=input_shape),
	# layers.Conv2D(8, kernel_size=(3, 3), kernel_regularizer=l1_l2(), activation="relu"),
	# layers.MaxPooling2D(pool_size=(2, 2)),
	# layers.Dropout(0.3),

	# layers.Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l1_l2(), activation="relu"),
	# layers.MaxPooling2D(pool_size=(2, 2)),
	# layers.Dropout(0.3),

	# layers.Flatten(),
	# layers.Dropout(0.5),
	# layers.Dense(len(labels), activation="softmax"),
	# ])

 	# ================ MODEL 5 ===========================

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


	#Creating a data generator that will retrieve
	#batches of images into training and 
	#validation sets that will be forwarded
	#to the model during training.
	datagen = ImageDataGenerator(
			rescale = 1. / 255,
			validation_split = 0.15,
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

	#Defining the model callbacks.
	class CallBack(tf.keras.callbacks.Callback):
		#Callbacks act as functions that get executed during
		#specific moments during training. 
		#Here they work as a way of early stopping the training
		#once the model has reached accuracy and loss within
		#a specified threshold.
		def on_epoch_end(self, epoch, logs={}):
			if((logs.get('accuracy') > Accuracy_Threshold) & (logs.get('loss') < loss_threshold) 
				& (logs.get('val_accuracy') > Accuracy_Threshold) & (logs.get('val_loss') < loss_threshold)):
				self.model.stop_training = True

	#Initiating the model callbacks.
	callbacks = CallBack()

	#The model will be compiled with the following parameters.
	model.compile(
		loss='categorical_crossentropy',
		optimizer = keras.optimizers.Adam(learning_rate = 0.0005), #rmsprop, adam, SGD
		metrics = ['accuracy']
	)


	GUI.theme('Light Blue 1')

	font = ('Arial', 13)

	#Creating a new GUI window.
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

	window = GUI.Window('OCR-Thesis Demo', layout)

	while True:
		#Reading events and values from the GUI window.
		event, values = window.read()

		#Closing the GUI window terminates the program.
		if event == GUI.WIN_CLOSED or event == 'Cancel':
			break

		#Pressing the button 'continue' changes the GUI window layout.
		if event == 'continue':
			window[f'-Training_Prompt-'].update(visible=False)
			window[f'-Insert_Epoch-'].update(visible=True)

		#Characters that are not digits will not appear in the input epoch box.
		if event == 'input' and values['input'] and values['input'][-1] not in ('0123456789'):
			window['input'].update(values['input'][:-1])

		#Pressing the 'ok' button to validate the inserted epoch
		#checks that the input is not empty and that it is not a 0
		if event == 'ok':
			if values['input'] != '':
				if values['input'] == '0':
					#Displaying popup warning if input is 0.
					GUI.popup('The training epochs may not be 0!', font=font, title='Error!')
				else:
					epochs = int(values['input'])

					#Changing to the -Training_In_Progress- layout
					window[f'-Insert_Epoch-'].update(visible=False)
					window[f'-Training_In_Progress-'].update(visible=True)

					#Initiating the training with the specified epochs.
					training_results = model.fit(ds_train, validation_data = ds_validate, epochs = epochs, verbose = 2, callbacks=[callbacks])
					print('\n\nTraining is Complete!')

			#Displaying popup warning if input is empty.
			else:
				GUI.popup('The input may not be empty!', font=font, title='Error!')

		#If buttons 'finish' or 'exit' have been pressed,
		#the program terminates.
		if ((event == 'finish') | (event == 'Exit')):
			break

	#Terminating the GUI Window.
	window.close()

	try:
		#Initializing the results graph creation.
		plot_training_results(training_results)
	except:
		return

	#Creating a new GUI Window
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

	window = GUI.Window('OCR-Thesis Demo', layout)

	model_name = ''

	#GUI Window existence loop.
	while True:
		#Reading GUI window events and values.
		event, values = window.read()
		
		#Closing the GUI window terminates the program.
		if event == GUI.WIN_CLOSED:
			break

		#Saving the model under the user-given name.
		if event == 'ModelSave':
			model_name = values['ModelName']
			#The name must not be empty.
			if (model_name != ''):
				#The name must not contain these invalid characters.
				if (re.search('[\\/:"*?<>|]+', model_name)):
					#Display GUI popup.
					GUI.popup('The name is invalid.', font=font, title='Error!')

				else:
					#Preparing the model and classes names.
					final_model_name = model_name + '.h5'
					classes_name = model_name + '.npy'
					#Moving directory over to the 'Models' folder.
					os.chdir(exec_directory_root + '/Models')
					model.save(final_model_name) #Saving the model.

					#Saving the model classes.
					classes = ds_train.class_indices
					model_classes = np.array(classes)
					save(classes_name, model_classes)

					#Changing the window scene to '-Model_Saved-'
					window[f'-Save_Model-'].update(visible=False)
					window[f'-Model_Saved-'].update(visible=True)

			else:
				#Display GUI popup.
				GUI.popup('The name may not be empty.', font=font, title='Error!')

		#Pressing the buttons 'Cancel' and 'Exit' closes
		#the GUI window and terminates the program.
		if ((event == 'Cancel') | (event == 'Exit')):
			break

		#Restarting the program.
		if event == 'model_save_finish':
			os.execv(sys.executable, ['python'] + sys.argv)

	#Terminating the GUI Window.
	window.close()



def Predict(images):
	global model
	global model_classes_from_file

	output_label = ''

	#Extracting the class names from the classes file.
	classes = np.load(model_classes_from_file, allow_pickle=True)
	classes_list = classes.tolist()
	classes_names = []
	for c in classes_list:
		classes_names.append(c)

	#Preparing the detected images for prediction.
	processed_images = ImagePreProcessing(images)

	#Predicting each character image and adding the result to a string.
	for pi in processed_images:
		prediction = model.predict(pi)
		classification = prediction.argmax(axis = -1)
		output_label = output_label + classes_names[int(classification)]

	return output_label


def LoadModel(model_path, classes_path):
	global model
	global model_classes_from_file

	#Loading model and storing the classes path.
	model = load_model(model_path)
	model_classes_from_file = classes_path



def ImagePreProcessing(images):

	processed_images = []

	for i in images:
		#Sharpening the images and then converting it back to BGR.
		gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)	# Converts the image to grayscale
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	# Inverts the image
		RGB_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		
		#Converting the images into an array and adding them to a list.
		img_arr = img_to_array(RGB_thresh)
		img_arr = img_arr / 255
		img_arr = np.expand_dims(img_arr, axis = 0)

		# cv2.imshow('Processed Image', RGB_thresh)	#Debug - Display processed image
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		processed_images.append(img_arr)

	return processed_images