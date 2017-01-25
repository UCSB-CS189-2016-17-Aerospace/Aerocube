import cv2 
import numpy as np
from skimage import data
from skimage import exposure 

class PreProcessor:

	threshold_value = 2.2

	def __init__(self, pathToNewImage):
		self.image = cv2.imread(pathToNewImage)
		self.scikitImage = data.imread(pathToNewImage)

	def is_similar(self, pathToOtherImage):
		existingImage = cv2.imread(pathToOtherImage)
		difference = cv2.subtract(self.image, existingImage)
		return np.mean(difference) <= self.threshold_value
		

	def calculate_average_exposure(self):
		pass

	def increase_contrast(self):
		pass

	def brighten_image(self):
		pass

	def is_low_contrast(self):
		isLow = exposure.is_low_contrast(self.scikitImage)
		print(isLow)