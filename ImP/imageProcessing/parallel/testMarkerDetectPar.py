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
        np.testing.assert_equal(MarkerDetectPar._threshold(self.gray, 4),
                                MarkerDetectPar._threshold(self.gray, 5))

    def test_threshold_returns_valid_thresholded_img(self):
        thresh = MarkerDetectPar._threshold(self.gray, 3)
        self.assertIsNotNone(thresh)
        self.assertIsNotNone(thresh.size)

    def test_threshold_equals_aruco_method(self):
        thresh_const = MarkerDetectPar.params[MarkerDetectPar.adaptiveThreshConstant]
        np.testing.assert_equal(MarkerDetectPar._threshold(self.gray, 3),
                                aruco._threshold(self.gray, 3, thresh_const))

    # PUBLIC FUNCTIONS

    def test_detector_parameters(self):
        pass

    def test_detect_markers_raise_on_improper_image(self):
        self.assertRaises(AssertionError,
                          MarkerDetectPar.detect_markers_parallel, None)

    def test_detect_candidates_raise_on_improper_image(self):
        self.assertRaises(AssertionError,
                          MarkerDetectPar._detect_candidates, self.image)
        self.assertRaises(AssertionError,
                          MarkerDetectPar._detect_candidates, None)

    # ~~STEP 1 FUNCTIONS~~

    def test_detect_initial_candidates_equals_aruco_method(self):
        # TODO: (14,) instead of (32,) being returned
        test_vals = MarkerDetectPar._detect_initial_candidates(self.gray)
        true_vals = aruco._detectInitialCandidates(self.gray)
        np.testing.assert_allclose(test_vals[0], true_vals[0])
        np.testing.assert_equal(test_vals[1], true_vals[1])

    def test_find_marker_contours_equals_aruco_method(self):
        """
        Tests find_marker_contours with various thresholded images.
        Note that candidate matrices are floats, and thus must be tested with allclose.
        However, contour matrices with ints should be tested with array_equal.
        :return:
        """
        # TODO: Assert equal for non-empty contours array fails for some reason
        params = MarkerDetectPar.params
        aruco_params = (params[MarkerDetectPar.minMarkerPerimeterRate],
                       params[MarkerDetectPar.maxMarkerPerimeterRate],
                       params[MarkerDetectPar.polygonalApproxAccuracyRate],
                       params[MarkerDetectPar.minCornerDistanceRate],
                       params[MarkerDetectPar.minDistanceToBorder])
        # thresh with winSize = 3
        thresh_3 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                         3, params[MarkerDetectPar.adaptiveThreshConstant])
        test_contours_thresh_3 = MarkerDetectPar._find_marker_contours(thresh_3)
        true_contours_thresh_3 = aruco._findMarkerContours(thresh_3, *aruco_params)
        np.testing.assert_allclose(test_contours_thresh_3[0], true_contours_thresh_3[0])
        np.testing.assert_array_equal(test_contours_thresh_3[1], true_contours_thresh_3[1])
        # thresh with winSize = 13
        thresh_13 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                          13, params[MarkerDetectPar.adaptiveThreshConstant])
        test_contours_thresh_13 = MarkerDetectPar._find_marker_contours(thresh_13)
        true_contours_thresh_13 = aruco._findMarkerContours(thresh_13, *aruco_params)
        np.testing.assert_allclose(test_contours_thresh_13[0], true_contours_thresh_13[0])
        np.testing.assert_array_equal(test_contours_thresh_13[1], true_contours_thresh_13[1])
        # thresh with winSize = 7
        thresh_7 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                         7, params[MarkerDetectPar.adaptiveThreshConstant])
        test_contours_thresh_7 = MarkerDetectPar._find_marker_contours(thresh_7)
        true_contours_thresh_7 = aruco._findMarkerContours(thresh_7, *aruco_params)
        np.testing.assert_allclose(test_contours_thresh_7[0], true_contours_thresh_7[0])
        np.testing.assert_array_equal(test_contours_thresh_7[1], true_contours_thresh_7[1])

    def test_assert_find_marker_contours_does_not_modify_thresh(self):
        params = MarkerDetectPar.params
        thresh = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       3, params[MarkerDetectPar.adaptiveThreshConstant])
        thresh_copy = np.copy(thresh)
        MarkerDetectPar._find_marker_contours(thresh)
        np.testing.assert_equal(thresh, thresh_copy)

    @unittest.skip("Aruco Python binding could be inadequate")
    def test_reorder_candidate_corners_equals_aruco_method(self):
        # TODO: Aruco call is failing; could try to rewrite Aruco function signature, but that seems risky
        candidates, _ = aruco._detectInitialCandidates(self.gray)
        aruco._reorderCandidatesCorners(candidates)
        np.testing.assert_array_equal(MarkerDetectPar._reorder_candidate_corners(candidates),
                                      aruco._reorderCandidatesCorners(candidates))

    # ~~STEP 2 FUNCTIONS~~

    # ~~STEP 3 FUNCTIONS~~

    # ~~STEP 4 FUNCTIONS~~

if __name__ == '__main__':
    unittest.main()
