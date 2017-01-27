import unittest 
import os 
from .preProcessor import PreProcessor 
from .testSettings import testSettings

class ImageSimilarityTestCase(unittest.TestCase):
	def setUp(self):
		self.directory = testSettings.get_test_files_path()
		self.pathForTestImage = os.path.join(self.directory,'im1.JPG')
		self.pathForNewImage = os.path.join(self.directory,'im5.JPG')
		self.pathForSimilarImage = os.path.join(self.directory,'im4.JPG')
		self.pathForDark = os.path.join(self.directory,'darkest.jpg')
		self.pathForSharpImage = os.path.join(self.directory,'im6.JPG')
		self.processor = PreProcessor(self.pathForTestImage)
		self.processor3 = PreProcessor(self.pathForSharpImage)
		self.processor2 = PreProcessor(self.pathForDark)

	def tearDown(self):
		
		self.processor.pilImage.close()
		self.processor2.pilImage.close()
		self.processor3.pilImage.close() #close images pil ones
		
		self.processor = None
		self.processor2 = None
		self.processor3 = None
	
	
	def test_identical_images(self):
		similarity = self.processor.is_similar(self.pathForTestImage)
		self.assertEqual(similarity, True)
	
	def test_different_images_fail(self):
		'''
			The idea is that the image used here is different enough for us to
			consider it a new image, or in other words has a difference value > 2.2

		'''
		similarity = self.processor.is_similar(self.pathForNewImage)
		self.assertEqual(similarity,False)

	def test_different_images_pass(self):
		'''
			Here the images should have a difference value <= 2.2, therefore it should pass
		'''
		similarity = self.processor.is_similar(self.pathForSimilarImage)
		self.assertEqual(similarity,True)
	
	def test_valid_image(self):
		'''
			This test simply checks to see that the return value of the cv2.imread() , which is used
			in is_similar() in preProcessor.py, is type ndarray. 

			This is because a file that is not found or not read correctly, is given type NoneType by
			cv2.imread()
		'''
		validaty = self.processor.image.__class__.__name__
		self.assertEqual(validaty,'ndarray')
	
	
	def test_create_darker_image(self):
		#check for darker image by comparing pixels maybe ?
		self.processor2.darken_image()
	
	def test_create_brighter_image(self):
		#check for darker image by comparing pixels maybe ?
		self.processor2.brighten_image()

	"""
	def test_is_low_contrast(self):
		#hard to set up test since we have yet to define a threshold for iamge contrast 
		self.processor2.is_low_contrast()
	"""

	"""
	def test_image_sharpening(self):
		self.processor3.sharper_image()
	"""

if __name__ == '__main__':
	unittest.main()
