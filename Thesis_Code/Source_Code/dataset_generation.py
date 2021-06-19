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
    font_name_array = ['arial', 'times', 'bahnschrift', 'cambria', 'constan', 'lucon', 'calibri', 'impact', 'segoepr', 'segoesc', 'comic']
    font_size = 35  # Font size of each alphanumeric
    letter_repetitions = 1 # Repetitions of letters that are hard to detect (Ex. AA, VV, etc.) - MAX 2, EXPONENTIAL

    # Unicode Greek Letter Codes = 902-974 (Includes a couple of empty codes!)
    # https://www.ssec.wisc.edu/~tomw/java/unicode.html#x0370

    letters = [] # Letter codes
    letters_in_image = []
    lowercase_letters = []  # Lowercase letter codes
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
    
    # if(selected_dataset == 'Special_chars'): # NOT CURRENTLY USED

    #     letters.append('dash')
    #     directory_letters.append('dash')
    #     letters_in_image.append(chr(45))

    #     letters.append('dot')
    #     directory_letters.append('dot')
    #     letters_in_image.append(chr(46))

    #     letters.append('slash')
    #     directory_letters.append('slash')
    #     letters_in_image.append(chr(47))

    #     letters.append('colon')
    #     directory_letters.append('colon')
    #     letters_in_image.append(chr(58))

    # ------------------------------------------------

    try:
        shutil.rmtree(dataset_directory_root)
    except:
        print('There was no dataset directory to reset - creating it now.')

    os.mkdir(dataset_directory_root)

    for l in range (0, len(letters)):  # For each character combination...

        counter = 1

        letter_directory = dataset_directory_root + '/' + directory_letters[l] # Setting up the directory path for each character
        os.mkdir(letter_directory)  # Creating the directories

        for font in font_name_array:  # And for each font on the font array

            picture_name = letters[l] + '_' + str(counter) + '.jpg'  # Setting up the picture name

            image = Image.new('RGB', (img_width,img_height)) # Create a new RGB image for every character with size x,y (PIL)

            draw = ImageDraw.Draw(image)  # Enable drawing on the image
            font = ImageFont.truetype(font, font_size)
            draw.text((10,int(img_height/(font_size/2))), letters_in_image[l], fill = (255,255,255), font = font)  # Draw the character on the image

            image = Augment_Image(image)

            os.chdir(letter_directory) # Moving the directory to the character folder
            image.save(picture_name)  # Using PIL to save the image

            counter = counter + 1

  
    # return

    TrainModel(dataset_directory_root, exec_directory_root, img_width, img_height)


def Augment_Image(image):

    cv2Image = np.array(image)
    cv2Image = cv2.cvtColor(cv2Image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh2 = thresh.copy()

    kernel = np.ones((13,13),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    
    cnts, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for c in cnts:

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


      # print(x, y, w, h)

      roi = thresh[y:y+h, x:x+w]

      # print(roi.shape())

      # cv2.imshow('test',roi)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()


      roi = cv2.resize(roi, (50, 50))

    cv2Image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(roi)

    return pilImage
