import cv2
import numpy as np
import imutils

from operator import itemgetter

from predict import Predict

original_image = [] # The original loaded image
current_image_state = [] # Marks the current state of the image with the selections. Used for 'undo' feature.
image_cache = [] # Before a rectangle gets drawn, a snapshot of the main image is stored as a cache. Used for 'undo' feature.
selection_images = [] # Selected rectangle images
filtered_selection_images = [] # Selected rectangle images without any images that couldn't be processed
processed_images = [] # Processed selection_images
selections = []	# Selected selection coordinates
drawPoints = []	# Starting and Ending coordinates of selection
selection_endpoint = []	# Ending coordinate of selection, used for dynamic drawing
drawing = False	# Default activity state



def selection(image_link):	# Main function which allows the selections to be drawn on the image

	global original_image
	global current_image_state
	global selection_images
	global selections
	global image_cache

	original_image = cv2.imread(image_link) # Read the image from the specified link
	original_image = imutils.resize(original_image, width = 1500)
	current_image_state = original_image.copy()

	cv2.namedWindow("Loaded Image",cv2.WINDOW_NORMAL)	# Create a window named "image"
	cv2.setMouseCallback("Loaded Image", draw_selection, current_image_state)	# Initializing the callback to draw the selections

	while True:
		if not drawing:
			cv2.imshow('Loaded Image', current_image_state)

		elif drawing and selection_endpoint:	# Dynamically draws the selection while the mouse is moving
			selection_copy = current_image_state.copy()
			cv2.rectangle(selection_copy, drawPoints[0], selection_endpoint[0], (0, 255, 0), 1)	# Draws from starting point
																								# to current end point
			
			cv2.imshow('Loaded Image', selection_copy)

		key = cv2.waitKey(1) & 0xFF

		if(key == 26): # ctr + z / ctr + Z for undo
			try:
				selections.pop()
				current_image_state = image_cache[-1]
				image_cache.pop()
			except:
				pass

		if ((key == ord("q")) | (key == ord("Q"))):	# Pressing "q" or "Q" closes the window
			break

	cv2.destroyAllWindows()


	Result = display_rectangles()
	return Result, filtered_selection_images



def draw_selection(event, x, y, flags, image): # Used in the callback to draw the selections on the image
	
	global drawPoints, drawing, selection_endpoint, selections
	global image_cache
	global current_image_state

	if event == cv2.EVENT_LBUTTONDOWN:
		drawPoints = [(x, y)]	# Mark the starting coordinates of the selection
		drawing = True	# Switching the drawing state to True

	elif event == cv2.EVENT_MOUSEMOVE & drawing:
		selection_endpoint = [(x, y)]	# While moving the mouse, add the current end coordinates

	elif event == cv2.EVENT_LBUTTONUP:
		drawPoints.append((x, y))	# Upon the release of the mouse button, mark the final end coordinates
		drawing = False	# Switching the drawing state back to False

		# print(drawPoints)	#Debug - Display coordinates of selection
		
		x = drawPoints[0][0]
		ix = drawPoints[1][0]
		y = drawPoints[0][1]
		iy = drawPoints[1][1]

		drawPoints = [(min(ix,x), min(iy,y)), (max(ix,x), max(iy,y))] # Making sure that the coordinates
																	  # are always in the correct order

		selections.append(drawPoints)	# Add the completed selection to the list

		image_cache.append(current_image_state.copy())

		cv2.rectangle(current_image_state, drawPoints[0], drawPoints[1], (0, 255, 0), 2)	# Draw the complete selection



def display_rectangles():	# Creates and displays a new image based on the selected area

	global selections, original_image, selection_images

	for s in range (0, len(selections)):	# Create a new image based on the selection coordinates
		selection = selections[s]
		selection_image = original_image[selection[0][1]:selection[1][1], selection[0][0]:selection[1][0]]
		try:
			# cv2.imshow('Selected Image',selection_image)	# Display the new selection image
			selection_images.append(selection_image)	# Store the image to a list
		except:
			pass

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	Result = image_processing()
	return Result



def image_processing():	# Processes the selected images #!!!!~~HEAVILY WORK IN PROGRESS~~!!!!
	
	global selection_images
	global filtered_selection_images

	for i in selection_images:

		try:
			gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)	# Converts the image to grayscale
			# blur = cv2.GaussianBlur(gray, (3,3), 0)	# Adds a soft blur to the image

			thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	# Inverts the image


			kernel = np.ones((15,1),np.uint8)

			closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


			processed_images.append(closing)	# Stores the processed images to a list

			# cv2.imshow('Processed Image', closing)	#Debug - Processed Image display

			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			filtered_selection_images.append(i)
		except:
			pass

	Result = contour_detection()
	return Result



def contour_detection(): # Detects contours inside the processed images

	global processed_images

	image_num = 0

	Result = ''

	for pi in processed_images:

		cnts, hierarchy = cv2.findContours(pi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detects contours

		Image_Result = contour_display(cnts, image_num)	# Sends the detected contours to contour_display with the image number

		image_num = image_num + 1

		# cv2.imshow('Contoured Image', pi) #Debug - Contoured Image display

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		if(Result == ''):
			Result = Result + Image_Result
		else:
			Result = Result + '\n' + Image_Result
	return Result

def contour_display(contours, image_num):	# Displays the detected contours on the image

	global filtered_selection_images

	rectangled_letters = []

	selection_img = filtered_selection_images[image_num]
	selection_img_copy = selection_img.copy()
	selection_img_copy = cv2.bitwise_not(selection_img_copy)
	
	sorted_rectangles = sort_contours(contours)

	totalWidth = 0

	for sr in sorted_rectangles:

		x = sr[0]
		y = sr[1]
		w = sr[2]
		h = sr[3]
		
		x = x-2
		y = y-2
		h = h+4
		w = w+4

		if((x-3) < 0):
			x = 0
		if((y-3) < 0):
			y = 0

		roi = selection_img[y:y+h, x:x+w]

		totalWidth = totalWidth + roi.shape[1]

	try:
		averageWidth = int(int(totalWidth / len(sorted_rectangles)) + (0.25 * int(totalWidth / len(sorted_rectangles)))) # Avg + 25% of Avg
	except:
		pass

	for sr in sorted_rectangles:

		x = sr[0]
		y = sr[1]
		w = sr[2]
		h = sr[3]
		
		x = x-2
		y = y-2
		h = h+4
		w = w+4

		if((x-3) < 0):
			x = 0
		if((y-3) < 0):
			y = 0


		try:

			roi = selection_img[y:y+h, x:x+w] # Create a smaller image to display each individual alphanumeric
			
			if (roi.shape[1] > averageWidth):

				roi_1 = selection_img[y:y+h, x:int(x + w/2)]
				roi_2 = selection_img[y:y+h, x + int(w/2):x + w]

				roi_1 = cv2.resize(roi_1, (50, 50))
				roi_2 = cv2.resize(roi_2, (50, 50))

				rectangled_letters.append(roi_1)
				rectangled_letters.append(roi_2)

			else:
				roi = cv2.resize(roi, (50, 50))
				rectangled_letters.append(roi)
		except:
			pass

	Result = Predict(rectangled_letters)
	return Result


	# cv2.imshow('Contours', selection_img)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def sort_contours(contours):

	contours_bound_rect = []

	for c in contours:
		(x,y,w,h) = cv2.boundingRect(c)

		contours_bound_rect.append([x,y,w,h])

	sorted_contours = sorted(contours_bound_rect, key=itemgetter(0))

	return sorted_contours