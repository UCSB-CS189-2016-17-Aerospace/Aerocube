import unittest
import os
import cv2
from ImP.imageProcessing.markerDetectionParallel import *
from ImP.imageProcessing.settings import ImageProcessingSettings


class TestMarkerDetectionParallel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._IMAGE = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'jetson_test1.jpg'))


    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.image = self._IMAGE

    def test_detect_candidates_raise_improper_image(self):
        self.assertRaises(MarkerDetectionParallel.CUDAFunctionException, MarkerDetectionParallel.detect_candidates, self.image)
        self.assertRaises(MarkerDetectionParallel.CUDAFunctionException, MarkerDetectionParallel.detect_candidates, None)

    def test_detector_parameters(self):
        pass

if __name__ == '__main__':
    unittest.main()

