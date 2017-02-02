import unittest
import os
import cv2
from cv2 import aruco
import numpy as np
from ImP.imageProcessing.parallel.markerDetectPar import *
from ImP.imageProcessing.settings import ImageProcessingSettings


class TestMarkerDetectPar(unittest.TestCase):

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

    # CUDA TEST FUNCTIONS

    def test_numba_jit_add(self):
        """
        Test Numba's JIT decorator is working.
        :return:
        """
        self.assertEqual(MarkerDetectPar.numba_jit_add(1, 2), 3)

    def test_cuda_increment_by_one(self):
        """
        Test Numba's CUDA JIT decorator is working.
        Note that array inputs to the CUDA function *must* be numpy arrays
        so that Numba knows how to properly translate it into CUDA code.
        :return:
        """
        # Initialize arr and copy of arr
        arr = np.array([1, 2, 3])
        print([x + 1 for x in arr])
        new_arr = np.copy(arr)
        # Launch the kernel
        threadsperblock = 32
        blockspergrid = (arr.size + threadsperblock - 1)
        MarkerDetectPar.cuda_increment_by_one[blockspergrid, threadsperblock](new_arr)
        print([x+1 for x in arr])
        print(new_arr)
        # Confirm results of Python and Cuda are equal
        self.assertTrue(np.array_equal([x+1 for x in arr], new_arr))

    # HELPER FUNCTIONS

    def test_threshold_raise_on_small_window(self):
        self.assertRaises(AssertionError,
                          MarkerDetectPar._threshold,
                          self.gray, 2)

    def test_threshold_winSize_adjusted_correctly(self):
        self.assertTrue(np.array_equal(MarkerDetectPar._threshold(self.gray, 4),
                                       MarkerDetectPar._threshold(self.gray, 5)))

    def test_threshold_returns_valid_thresholded_img(self):
        thresh = MarkerDetectPar._threshold(self.gray, 3)
        self.assertIsNotNone(thresh)
        self.assertIsNotNone(thresh.size)

    def test_threshold_equals_aruco_threshold(self):
        thresh_const = MarkerDetectPar.detectorParams[MarkerDetectPar.adaptiveThreshConstant]
        self.assertTrue(np.array_equal(MarkerDetectPar._threshold(self.gray, 3),
                                       aruco._threshold(self.gray, 3, thresh_const)))

    # PUBLIC FUNCTIONS

    def test_detector_parameters(self):
        pass

    def test_detect_markers_raise_on_improper_image(self):
        self.assertRaises(MarkerDetectPar.MarkerDetectParException,
                          MarkerDetectPar.detect_markers_parallel, None)

    def test_detect_candidates_raise_on_improper_image(self):
        self.assertRaises(MarkerDetectPar.MarkerDetectParException,
                          MarkerDetectPar._detect_candidates, self.image)
        self.assertRaises(MarkerDetectPar.MarkerDetectParException,
                          MarkerDetectPar._detect_candidates, None)

    # ~~STEP 1 FUNCTIONS~~

    def test_find_marker_contours(self):
        pass

    # ~~STEP 2 FUNCTIONS~~

    # ~~STEP 3 FUNCTIONS~~

    # ~~STEP 4 FUNCTIONS~~

if __name__ == '__main__':
    unittest.main()
