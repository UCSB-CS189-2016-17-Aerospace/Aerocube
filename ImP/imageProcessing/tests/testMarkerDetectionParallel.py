import unittest
import os
import cv2
from ImP.imageProcessing.markerDetectionParallelWrapper import *
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

    def tearDown(self):
        self.image = None

    def test_detect_markers_raise_on_improper_image(self):
        self.assertRaises(MarkerDetectionParallelWrapper.MarkerDetectionParallelException,
                          MarkerDetectionParallelWrapper.detect_markers_parallel, None)

    def test_detect_candidates_raise_on_improper_image(self):
        self.assertRaises(MarkerDetectionParallelWrapper.MarkerDetectionParallelException,
                          MarkerDetectionParallelWrapper._detect_candidates, self.image)
        self.assertRaises(MarkerDetectionParallelWrapper.MarkerDetectionParallelException,
                          MarkerDetectionParallelWrapper._detect_candidates, None)

    def test_detector_parameters(self):
        pass

if __name__ == '__main__':
    unittest.main()

