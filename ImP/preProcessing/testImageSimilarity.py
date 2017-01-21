import unittest 
from preProcessor import PreProcessor 

class ImageSimilarityTestCase(unittest.TestCase):
	def setUp(self):
		self.pathForTestImage = 'test_pictures/im1.JPG'
		self.processor = PreProcessor(self.pathForTestImage)

	
	def test_identical_images(self):
		similarity = self.processor.is_similar(self.pathForTestImage)
		self.assertEqual(similarity, True)
	
	def test_different_images(self):
		'''
			The idea is that the image used here is different enough for us to
			consider it a new image, or in other words has a difference value > 2.2

		'''
		pathForNewImage = 'test_pictures/im5.JPG'
		similarity = self.processor.is_similar(pathForNewImage)
		self.assertEqual(similarity,False)



if __name__ == '__main__':
	unittest.main()