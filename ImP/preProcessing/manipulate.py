# IGNORE FILE
# FILE WAS ONLY USED TO PROTOTYPE



import cv2 
import numpy as np
from skimage import data
from skimage import exposure 
from PIL import Image, ImageFilter
from .testSettings import testSettings
import os	
from .preProcessor import PreProcessor 


directory = testSettings.get_test_files_path()	
pathForTestImage = os.path.join(directory,'brighter.jpg')

processor = PreProcessor(pathForTestImage)
temp = processor.pilImage.copy()
#processor.pilImage.save('taco.jpg')
processor.brighten_image(1.5)
#temp.save(processor.path)

