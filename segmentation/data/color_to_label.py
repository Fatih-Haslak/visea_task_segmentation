from PIL import Image
import numpy as np
import cv2
import os
def color_image_to_labelmap(image):#RGB İMAGE
 
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            renk = tuple(image[row, column])
            
            if renk == (142, 0, 0): #Car
                image[row, column] = 1
            elif renk == (60, 20, 220): #human
                image[row, column] = 2
            elif renk == (0,0,255): #human
                image[row,column] = 2
            else:
                image[row, column] = 0


