import cv2 
import numpy as np
from skimage import data
from skimage import exposure 
from PIL import Image


class PreProcessor:
    """
    TODO: Perhaps make a new task to develop a "main" function that takes the newly taken image and takes it through a
     process that first runs image similarity, then the low/high contrast check, and finally adjust the image until it passes the low/high
     contrast check.
    """
    image_similarity_threshold = 2.2

    def __init__(self, pathToNewImage):
        self.image = cv2.imread(pathToNewImage) # for OpenCV
        self.scikitImage = data.imread(pathToNewImage) # for scikit
        self.pilImage = Image.open(pathToNewImage) # for PIL
        self.path = pathToNewImage # used to save modified images to the same location, overriding them essentially

    def is_similar(self, pathToOtherImage):
        existingImage = cv2.imread(pathToOtherImage)
        difference = cv2.subtract(self.image, existingImage)
        self.pilImage.close()
        return np.mean(difference) <= self.image_similarity_threshold

    def darken_image(self, darkFactor):
        """
        By setting the darkfactor to anything less than 1.0 we can darken the image.
        """
        self.pilImage.point(lambda x: x*darkFactor).save(self.path)

    def brighten_image(self, brightFactor):
        """
        Works the same way as darken_image() but x>1.0 to brighten the image as opposed to
        being x<1.0 to darken it.
        """
        self.pilImage.point(lambda x: x*brightFactor).save(self.path)

    def is_low_contrast(self, threshold):
        """
        By default the threshold value is set to 0.05. We override that amount by setting our own threshold.

        TODO: Find threshold we will use.
        """
        return exposure.is_low_contrast(self.scikitImage, fraction_threshold=threshold)

    def is_too_bright(self, threshold):
        """
        Here we simply use a higher threshold to inverse the use of the exposure.is_low_contrast and detect
        high contrast.

        TODO: Find threshold we will use.
        """
        return exposure.is_low_contrast(self.scikitImage, fraction_threshold=threshold)


