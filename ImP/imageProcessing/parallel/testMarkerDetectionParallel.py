import unittest
import os
import cv2
from cv2 import aruco
import numpy as np

from ImP.imageProcessing.parallel.markerDetectionParallelWrapper import *
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
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def tearDown(self):
        self.image = None

    # HELPER FUNCTIONS

    def test_threshold_raise_on_small_window(self):
        self.assertRaises(AssertionError,
                          MarkerDetectionParallelWrapper._threshold,
                          self.gray, 2)

    def test_threshold_winSize_adjusted_correctly(self):
        self.assertTrue(np.array_equal(MarkerDetectionParallelWrapper._threshold(self.gray, 4),
                                       MarkerDetectionParallelWrapper._threshold(self.gray, 5)))

    def test_threshold_returns_valid_thresholded_img(self):
        thresh = MarkerDetectionParallelWrapper._threshold(self.gray, 3)
        self.assertIsNotNone(thresh)
        self.assertIsNotNone(thresh.size)

    def test_threshold_equals_aruco_threshold(self):
        thresh_const = MarkerDetectionParallelWrapper.detectorParams[MarkerDetectionParallelWrapper.adaptiveThreshConstant]
        self.assertTrue(np.array_equal(MarkerDetectionParallelWrapper._threshold(self.gray, 3),
                                       aruco._threshold(self.gray, 3, thresh_const)))


    # PUBLIC FUNCTIONS

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

