import unittest 
import os 
from .preProcessor import PreProcessor 
from .testSettings import testSettings

class ImageSimilarityTestCase(unittest.TestCase):
	def setUp(self):
		self.directory = testSettings.get_test_files_path()
		self.pathForTestImage = os.path.join(self.directory,'im1.JPG')
		self.pathForNewImage = os.path.join(self.directory,'im5.JPG')
		self.processor = PreProcessor(self.pathForTestImage)
	
	def tearDown(self):
		self.processor = None
	
	
	def test_identical_images(self):
		similarity = self.processor.is_similar(self.pathForTestImage)
		self.assertEqual(similarity, True)
	
	def test_different_images(self):
		'''
			The idea is that the image used here is different enough for us to
			consider it a new image, or in other words has a difference value > 2.2

		'''
		similarity = self.processor.is_similar(self.pathForNewImage)
		self.assertEqual(similarity,False)
	


if __name__ == '__main__':
	unittest.main()