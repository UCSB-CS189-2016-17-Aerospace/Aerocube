import scipy as sp 
from scipy.misc import imread
from scipy.signal.signaltools import correlate2d as c2d
import cv2 
import numpy as np

# below is the original approach for finding the difference, took too long, likely won't use
"""
def get(x):
	data = imread('distorted%s.jpg' % x)

	data = sp.inner(data, [299, 587, 114]) / 1000.0 

	return (data - data.mean()) / data.std()

im1 = get(1)
#im5 = get(5)
#print(im1.shape)

#im2 = get(2)
#im3 = get(3)
#im4 = get(4)

c11 = c2d(im1, im1, mode='same')
print(c11.max())
"""
#new approach that is much faster but does not directly quantify data in single number format
image1 = cv2.imread("im1.jpg")
image2 = cv2.imread("im1.jpg")

difference = cv2.subtract(image1, image2) #returns type numpy.ndarray


result = not np.any(difference)

if result is True:
	print("they are the same")
else:
	cv2.imwrite("result.jpg",difference)
	print("they are not the same")
