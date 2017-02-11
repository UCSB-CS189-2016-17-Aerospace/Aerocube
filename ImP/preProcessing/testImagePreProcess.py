import unittest 
import os 
from .preProcessor import PreProcessor 
from .testSettings import testSettings


class ImagePreProcessTestCase(unittest.TestCase):
    def setUp(self):
        """
        pathForTestImage is for image similarity

        pathForNewImage is for image similarity

        pathForSimilarImage is for image similarity

        self.processor is for all of these

        -------

        pathForDark is for darken_image testcase,

        self.processor2 can be used for both, so same image essentially

        """
        self.directory = testSettings.get_test_files_path()
        self.pathForTestImage = os.path.join(self.directory,'im1.JPG')
        self.pathForNewImage = os.path.join(self.directory,'im5.JPG')
        self.pathForSimilarImage = os.path.join(self.directory,'im4.JPG')
        self.pathForDark = os.path.join(self.directory,'brighter.jpg')

        self.processor = PreProcessor(self.pathForTestImage)
        self.processor2 = PreProcessor(self.pathForDark)
        self.temp = self.processor2.pilImage.copy()  # temp variable used to store copy of image so we can restore it after modifying it

    def tearDown(self):
        self.temp.save(self.processor2.path)
        self.processor.pilImage.close()
        self.processor2.pilImage.close()
        self.processor = None
        self.processor2 = None

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
        """
            Here the images should have a difference value <= 2.2, therefore it should pass
        """
        similarity = self.processor.is_similar(self.pathForSimilarImage)
        self.assertEqual(similarity,True)

    def test_valid_image(self):
        """
            This test simply checks to see that the return value of the cv2.imread() , which is used
            in is_similar() in preProcessor.py, is type ndarray.

            This is because a file that is not found or not read correctly, is given type NoneType by
            cv2.imread()
        """
        validaty = self.processor.image.__class__.__name__
        self.assertEqual(validaty,'ndarray')

    def test_create_darker_image(self):
        # not so sure how to test if the image is darker w/out invoking the methods used to test low/high contrast
        self.processor2.darken_image(0.6)

    def test_create_brighter_image(self):
        # not so sure how to test if the image is brighter w/out invoking the methods used to test low/high contrast
        self.processor2.brighten_image(1.4)

    def test_is_low_contrast(self):
        """
        No actual test set up since we do not have a current set threshold for low contrast
        """
        self.processor2.is_low_contrast(0.24)

    def test_too_bright(self):
        """
        No actual test set up since we do not have a current set threshold for low contrast
        """
        self.processor2.is_too_bright(0.9)


if __name__ == '__main__':
    unittest.main()
