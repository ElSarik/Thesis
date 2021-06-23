import os
import shutil

import cv2
import numpy as np
from numpy import save

import itertools

from predict import TrainModel


from PIL import Image, ImageFont, ImageDraw


def generate_store_dataset(selected_dataset):


    exec_directory_root = os.path.dirname(os.path.realpath(__file__)) #Program execution location
    dataset_directory_root = os.path.dirname(os.path.realpath(__file__)) + '/Dataset' #Dataset generation location

    img_width = 50  # x,y size of generated images
    img_height = 50
    font_name_array = ['arial', 'times', 'bahnschrift', 'cambria', 'constan', 
                       'lucon', 'calibri', 'impact', 'segoepr', 'segoesc', 'comic',
                       'pala', ] #List of fonts used
    font_size = 35  # Font size of each alphanumeric

    # Unicode Greek Letter Codes = 902-974 (Includes a couple of empty codes!)
    # https://www.ssec.wisc.edu/~tomw/java/unicode.html#x0370

    letters = [] # Letter codes
    letters_in_image = []
    directory_letters = []

    # ----------------LATIN CAPITAL----------------------

    if(selected_dataset == 'Latin'):
        for l in range (65, 91):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

    # ---------------LATIN CAPITAL & NUMBERS-------------

    if(selected_dataset == 'Latin_Nums'):
        for l in range (65, 91):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

        for l in range (48, 58):  # Numbers
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

    # ----------------GREEK CAPITAL----------------------

    if(selected_dataset == 'Greek'):
        for l in range (913, 930):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

        for l in range (931, 938):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

    # ---------------GREEK CAPITAL & NUMBERS-------------

    if(selected_dataset == 'Greek_Nums'):
        for l in range (913, 930):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

        for l in range (931, 938):
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

        for l in range (48, 58):  # Numbers
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

    # ----------------NUMBERS---------------------------

    if(selected_dataset == 'Nums'):
        for l in range (48, 58):  # Numbers
            letters.append(chr(l))
            directory_letters.append(chr(l))
            letters_in_image.append(chr(l))

    # ------------------------------------------------
    

    try:
        #Checking if a database folder exists.
        shutil.rmtree(dataset_directory_root)
    except:
        print('There was no dataset directory to reset - creating it now.')

    #Creating the database folder.
    os.mkdir(dataset_directory_root)

    # For each character combination...
    for l in range (0, len(letters)):

        counter = 1

        # Setting up the directory path for each character
        letter_directory = dataset_directory_root + '/' + directory_letters[l]
        
        # Creating the directories
        os.mkdir(letter_directory)

        # And for each font on the font array
        for font in font_name_array:
            try:
                font = ImageFont.truetype(font, font_size)

                # Setting up the picture name
                picture_name = letters[l] + '_' + str(counter) + '.jpg'

                # Create a new RGB image for every character with size x,y (PIL)
                image = Image.new('RGB', (img_width,img_height))

                # Enable drawing on the image
                draw = ImageDraw.Draw(image)

                # Draw the character on the image
                draw.text((10,int(img_height/(font_size/2))), letters_in_image[l], 
                fill = (255,255,255), font = font)

                image = Augment_Image(image)

                # Moving the directory to the character folder
                os.chdir(letter_directory)

                # Using PIL to save the image
                image.save(picture_name)

                counter = counter + 1
            except:
                pass


    TrainModel(dataset_directory_root, exec_directory_root, img_width, img_height)

#Formating and centering the character images
def Augment_Image(image):

    cv2Image = np.array(image)
    cv2Image = cv2.cvtColor(cv2Image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh2 = thresh.copy()

    kernel = np.ones((13,13),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #Detecting contours inside the images.
    cnts, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for c in cnts:

      #Bounding the contours into a rectangle, then expanding the rectangle.
      (x, y, w, h) = cv2.boundingRect(c)
      x = x-3
      y = y-3
      w = w+6
      h = h+6

      if((x-3) < 0):
          x = 0
      if((y-3) < 0):
          y = 0
      
      cv2.rectangle(thresh2, (x, y), (x + w, y + h), (255, 0, 0), 1)

      #Creating a new image based on the expanded rectangle
      roi = thresh[y:y+h, x:x+w]

      # cv2.imshow('Image',roi) #Debug - Displaying edited image.
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

      #Resizing the new image.
      roi = cv2.resize(roi, (50, 50))

    cv2Image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(roi)

    return pilImage
