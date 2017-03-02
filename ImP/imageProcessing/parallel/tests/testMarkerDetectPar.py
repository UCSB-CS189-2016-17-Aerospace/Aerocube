import os
import unittest
import cv2
import numpy as np
from ImP.imageProcessing.parallel.markerDetectParGold import MarkerDetectPar as MarkerDetectParGold
import ImP.imageProcessing.parallel.markerDetectPar as MarkerDetectPar
from ImP.imageProcessing.settings import ImageProcessingSettings


class TestMarkerDetectPar(unittest.TestCase):
    """
    Concerned with testing the accuracy/correctness of markerDetectPar with respect to markerDetectParGold (which
    should be verified as being essentially identical to the Aruco detectMarker method).
    """
    @classmethod
    def setUpClass(cls):
        cls._CAPSTONE_PHOTO_DIR = os.path.join(ImageProcessingSettings.get_test_files_path(),
                                               'capstone_class_photoshoot')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detect_markers_parallel_on_test_files(self):
        possible_files = [os.path.join(ImageProcessingSettings.get_test_files_path(), f) for f in os.listdir(ImageProcessingSettings.get_test_files_path())]
        img_paths = [f for f in possible_files if os.path.isfile(f)]
        for img_path in img_paths:
            img = cv2.imread(img_path)
            actual_corners, actual_ids = MarkerDetectPar.detect_markers_parallel(img)
            expected_corners, expected_ids = MarkerDetectParGold.detect_markers_parallel(img)
            np.testing.assert_allclose(actual_corners, expected_corners)
            np.testing.assert_array_equal(actual_ids, expected_ids)
            print("PASSED: {}".format(img_path))

    def test_detect_markers_parallel_on_capstone_photos(self):
        img_paths = [os.path.join(self._CAPSTONE_PHOTO_DIR, f) for f in os.listdir(self._CAPSTONE_PHOTO_DIR) if os.path.isfile(os.path.join(self._CAPSTONE_PHOTO_DIR, f))]
        for img_path in img_paths:
            img = cv2.imread(img_path)
            actual_corners, actual_ids = MarkerDetectPar.detect_markers_parallel(img)
            expected_corners, expected_ids = MarkerDetectParGold.detect_markers_parallel(img)
            np.testing.assert_allclose(actual_corners, expected_corners)
            np.testing.assert_array_equal(actual_ids, expected_ids)
            print("PASSED: {}".format(img_path))
