# LIST OF PANGRAMS FOR MULTI LETTER TESTING:
# http://clagnut.com/blog/2380/

import os
import os.path
from os import path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import webbrowser

from detect import selection
from dataset_generation import generate_store_dataset

from predict import LoadModel

from tensorflow.keras.preprocessing import image

import cv2

import PySimpleGUI as GUI

import re

from PIL import Image
import base64
from io import BytesIO

def main():
	Result = ''
	selected_images = []

	Image_file_types = [("JPEG (*.jpg)", "*.jpg"),
						("PNG (*.png)", "*.png")]

	Model_file_types = [("H5 (*.h5)", "*.h5")]

# -----------------------------------------------------------------------------------

	GUI.theme('Light Blue 1')

	font = ('Arial', 13)

	disclaimer_R1 = [[GUI.Text('DISCLAIMER:', font=font)]]
	disclaimer_R2 = [[GUI.Text('The following software was developed as a Demo to accompany my Bachelor Thesis:\n'
						 	   '"Development of an Optical Character Recognition application for alphanumeric data\n'
						 	   'using machine learning techniques".', font=font)],
				  	 [GUI.Text('')],
				  	 [GUI.Text('The software was designed to work best with images featuring typed, capital characters.\n'
				  	 		   'Any other type of images may produce unexpected results during the recognition.', font=font)],
				  	 [GUI.Text('')],
				  	 [GUI.Text('It was written to the best of my knowledge, though I can not guarantee that it\n'
						 	   'is completely free of errors.', font=font)],
				  	 [GUI.Text('')],
				  	 [GUI.Text('The source code can be found here:', font=font), GUI.Text('GitHub link', tooltip='https://github.com/ElSarik/Bachelor_Thesis_OCR', enable_events=True, font=('Arial', 13, 'underline'), key=f'URL')],
				  	 [GUI.Text('')],
				  	 [GUI.Text('University of Ioannina - Department of Computer Engineering - Arta, Greece', font=font)],
				  	 [GUI.Text('© Sarikakis Elias - Ioannes, 2021\n'
						 	   'All rights reserved.', font=font)]]
	disclaimer_R3 = [[GUI.Button('Continue', key='disclaimer_continue', font=font)]]
	disclaimer = [[GUI.Col([
							[GUI.Frame(layout = disclaimer_R1, title='', border_width = 0)],
				  			[GUI.Frame(layout = disclaimer_R2, title='', border_width = 0)],
				  			[GUI.Frame(layout = disclaimer_R3, title='', border_width = 0)]
				  		   ], element_justification = 'center')]]

	# ------------------------------------------------------

	load_create_R1 = [[GUI.Text('You are able to load an existing Model from your computer. For example a \n'
								'pre-trained Model included with this software.', font=font)],
					 [GUI.Text('Or you may go through the process of creating your own Model.', font=font)],
					 [GUI.Text('Please note that the complexity of the model can only be changed from\n'
					 		   'the source code.', font=font)]]
	load_create_R2 = [[GUI.Button('Load a Model', key='load_model', font=font, pad=(15,0)), GUI.Button('Create a Model', key='create_model', font=font, pad=(15,0))]]
	load_create = [[GUI.Col([
							 [GUI.Frame(layout = load_create_R1, title = '', border_width = 0)],
							 [GUI.Frame(layout = load_create_R2, title = '', border_width = 0)]
							], element_justification = 'center')]]
	# ------------------------------------------------------

	select_model = [[GUI.Text('Select a Model', font=font)],
					[GUI.Input(key='ModelPath'), GUI.FileBrowse(file_types=Model_file_types)],
					[GUI.OK(key='load_model_selected', font=font, pad=(15,0)), GUI.Cancel(key='load_model_cancel', font=font, pad=(15,0))]]

	# ------------------------------------------------------

	select_image = [[GUI.Text('Select an Image', font=font)],
					[GUI.Input(key='ImagePath'), GUI.FileBrowse(file_types=Image_file_types)],
					[GUI.OK(key='load_image_selected', font=font, pad=(15,0)), GUI.Cancel(key='load_image_cancel', font=font, pad=(15,0))]]

	select_dataset_text = [[GUI.Text('Select one of the datasets below.', font=font)],
					  	   [GUI.Text('Please note that datasets with both characters and numbers can be very\n'
					  				 'unpredictable during the prediction process.\n'
					  				 'For example an O can be predicted as a 0 (zero).', font=font)],
					  	   [GUI.Radio('Latin (capital)', 'dataset', True, key='Latin', font=font),
					   		GUI.Radio('Latin (capital) + Numbers', 'dataset', key='Latin_Nums', font=font),
					   		GUI.Radio('Numbers', 'dataset', key='Nums', font=font)],
					  	   [GUI.Radio('Greek (capital)', 'dataset', key='Greek', font=font),
					   		GUI.Radio('Greek (capital) + Numbers', 'dataset', key ='Greek_Nums', font=font)]]
	select_dataset_buttons = [[GUI.Button('Start Training', key='start_training', font=font, pad=(15,0)), GUI.Button('Cancel', key = 'cancel_training', font=font, pad=(15,0))]]
	select_dataset = [[GUI.Col([
								[GUI.Frame(layout = select_dataset_text, title = '', border_width = 0)],
								[GUI.Frame(layout = select_dataset_buttons, title = '', border_width = 0)]
								], element_justification = 'center')]]

	# ------------------------------------------------------

	layout = [[GUI.Column(disclaimer, visible=True, key='-Disclaimer-'),
			   GUI.Column(load_create, visible=False, key='-Load_Create-'),
			   GUI.Column(select_model, visible=False, key='-Select_Model-'),
			   GUI.Column(select_image, visible=False, key='-Select_Image-'),
			   GUI.Column(select_dataset, visible=False, key='-Select_Dataset-')]]


	window = GUI.Window('Optical Character Recognition Demo', layout)

	dataset = ''

	while True:
	    event, values = window.read()
	    
	    if event == GUI.WIN_CLOSED or event == 'Cancel':
	        break

		# -----------------Disclaimer-----------------------

	    if event == 'disclaimer_continue':
	    	window[f'-Disclaimer-'].update(visible=False)
	    	window[f'-Load_Create-'].update(visible=True)

	    if event == 'URL':
	    	webbrowser.open('https://github.com/ElSarik/Bachelor_Thesis_OCR')

		# ----------------Load_Create--------------------------

	    if event == 'load_model':
	    	window[f'-Load_Create-'].update(visible=False)
	    	window[f'-Select_Model-'].update(visible=True)
	    
	    if event == 'create_model':
	    	window[f'-Load_Create-'].update(visible=False)
	    	window[f'-Select_Dataset-'].update(visible=True)

		# -----------------Select_Model--------------------------
		
	    if event == 'load_model_selected':
	    	model_path = values['ModelPath']
	    	if (re.search('\/.*?\.[\w:]+', model_path)):
	    		if(path.exists(model_path)):
	    			window[f'-Select_Model-'].update(visible=False)
	    			window[f'-Select_Image-'].update(visible=True)

	    			x = model_path.split('.h5')[0]
	    			classes_path = x + '.npy'

	    			LoadModel(model_path, classes_path)
	    		else:
	    			GUI.popup('The model cannot be found on the specified path.', font=font)
	    	else:
	    		GUI.popup('The path seems to be invalid.', font=font)

	    if event == 'load_model_cancel':
	    	window[f'-Select_Model-'].update(visible=False)
	    	window[f'-Load_Create-'].update(visible=True)

		# ------------------Select_Image------------------------

	    if event == 'load_image_selected':
	    	image_path = values['ImagePath']
	    	if (re.search('\/.*?\.[\w:]+', image_path)):
	    		if(path.exists(image_path)):
	    			GUI.popup('Both the model and the image have been successfully loaded!\n\n'
	    					  'A new window will now open where you will be able to create regions '
	    					  'on the image where the model will later attempt\n' 
	    					  'to recognise its contents.\n\n'
	    					  'INSTRUCTIONS:\n'
	    					  '- PRESS, HOLD and MOVE your LEFT MOUSE CLICK\n'
	    					  'to create a region for the model on top of the image.\n'
	    					  '- Press "CTRL + Z" to undo a created region.\n'
	    					  '- Press "Q" to finalize your selection and proceed with the recognition.\n'
	    					  '- You may also resize the window if needed.\n\n'
	    					  'Please not that the window will stay open until the program is either killed '
	    					  'or the "Q" key has been pressed!', font=font)
	    			window[f'-Select_Image-'].update(visible=False)
	    			Result, selected_images = selection(image_path)
	    			break
	    		else:
	    			GUI.popup('The image cannot be found on the specified path.', font=font)
	    	else:
	    		GUI.popup('The path seems to be invalid.', font=font)

	    if event == 'load_image_cancel':
	    	window[f'-Select_Image-'].update(visible=False)
	    	window[f'-Select_Model-'].update(visible=True)

		# ------------------Select_Dataset----------------------

	    if event == 'start_training':
	    	if values['Latin']:
	    		dataset = 'Latin'
	    		window.close()
	    		break

	    	if values['Latin_Nums']:
	    		dataset = 'Latin_Nums'
	    		window.close()
	    		break

	    	if values['Greek']:
	    		dataset = 'Greek'
	    		window.close()
	    		break

	    	if values['Greek_Nums']:
	    		dataset = 'Greek_Nums'
	    		window.close()
	    		break

	    	if values['Nums']:
	    		dataset = 'Nums'
	    		window.close()
	    		break

	    if event == 'cancel_training':
	    	window[f'-Select_Dataset-'].update(visible=False)
	    	window[f'-Load_Create-'].update(visible=True)

	if(dataset != ''):
		generate_store_dataset(dataset)

	window.close()


	Results = Result.split('\n')

	if(Result != ''):

		results_layout = []

		for i in range(0, len(selected_images)):
					
			buffered = BytesIO()	# Creating a bytes buffer to convert from nparray to base64

			try:
				converted_Image = Image.fromarray(selected_images[i], 'RGB') # Converting from nparray to Image.
				converted_Image.save(buffered, format="PNG")	# Saving the image to a file-like object as PNG.
				Image_binary = buffered.getvalue()	#  Converting the PNG image to a binary format.

				base64_image = base64.b64encode(Image_binary) # Converting the binary image to base64

				if i != 0:
				    results_layout += GUI.Column([[GUI.Frame(layout = [[GUI.Column([[GUI.Text(f'Image {i+1} of {len(selected_images)}', font=font)],[GUI.Image(data = base64_image)], [GUI.Text(Results[i], font=font)]]),]], title = '', border_width = 0, element_justification = 'center'),]], visible = False, key = f'-Result_{i}-'),
				else:
				    results_layout += GUI.Column([[GUI.Frame(layout = [[GUI.Column([[GUI.Text(f'Image {i+1} of {len(selected_images)}', font=font)],[GUI.Image(data = base64_image)], [GUI.Text(Results[i], font=font)]]),]], title = '', border_width = 0, element_justification = 'center'),]], visible = True, key = f'-Result_{i}-'),
			except:
				pass

		results_layout = [results_layout]
		results_layout += [[GUI.Column([
										[GUI.Button('Previous Result', font=font), 
										 GUI.Button('Next Result', font=font), 
										 GUI.Button('Finish', font=font)]], 
							element_justification = 'center')]]

		final_results_layout = [[GUI.Column(results_layout, element_justification = 'center')]]

		window = GUI.Window('RESULTS', final_results_layout)


		current_layout = 0

		while True:
		    event, values = window.read()
		    
		    if event == GUI.WIN_CLOSED or event == 'Finish':
		        
		        break

		    if event == 'Next Result':

		        window[f'-Result_{current_layout}-'].update(visible=False)
		        if ((current_layout + 1) == len(selected_images)):
		        	current_layout = 0
		        	window[f'-Result_{current_layout}-'].update(visible=True)
		        else:
		        	current_layout = current_layout + 1
		        	window[f'-Result_{current_layout}-'].update(visible=True)

		    if event == 'Previous Result':

		        window[f'-Result_{current_layout}-'].update(visible=False)
		        if ((current_layout - 1) < 0):
		        	current_layout = (len(selected_images) - 1)
		        	window[f'-Result_{current_layout}-'].update(visible=True)
		        else:
		        	current_layout = current_layout - 1
		        	window[f'-Result_{current_layout}-'].update(visible=True)

		window.close()


if __name__=="__main__":

	main()
