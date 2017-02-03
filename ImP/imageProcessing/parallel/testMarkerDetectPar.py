import unittest
import os
import cv2
from cv2 import aruco
import numpy as np
from ImP.imageProcessing.aerocubeMarker import AeroCubeMarker
from ImP.imageProcessing.parallel.markerDetectPar import *
from ImP.imageProcessing.settings import ImageProcessingSettings


class TestMarkerDetectPar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._IMAGE = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'jetson_test1.jpg'))
        cls._IMG_MARKER = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0.png'))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.image = np.copy(self._IMAGE)
        self.img_marker = np.copy(self._IMG_MARKER)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray_marker = cv2.cvtColor(self.img_marker, cv2.COLOR_BGR2GRAY)

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

    def test_detect_candidates_equals_aruco_method(self):
        candidates, contours = MarkerDetectPar._detect_candidates(self.gray)
        aruco_cand, aruco_cont = aruco._detectCandidates(self.gray, aruco.DetectorParameters_create())
        np.testing.assert_allclose(candidates, aruco_cand)
        np.testing.assert_array_equal(contours, aruco_cont)

    def test_detect_initial_candidates_equals_aruco_method(self):
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

    def test_reorder_candidate_corners_preserves_shape_and_alters_param(self):
        candidates, _ = aruco._detectInitialCandidates(self.gray)
        cand_copy = np.copy(candidates)
        tmp = MarkerDetectPar._reorder_candidate_corners(candidates)
        self.assertFalse(np.allclose(cand_copy, candidates))
        self.assertFalse(np.allclose(cand_copy, tmp))
        self.assertEqual(np.array(candidates).shape, cand_copy.shape)

    @unittest.skip("Aruco Python binding failing")
    def test_filter_too_close_candidates_equals_aruco_method(self):
        candidates, contours = aruco._detectInitialCandidates(self.gray)
        # Use own method to simulate reordering of corners, since Aruco function is not working
        MarkerDetectPar._reorder_candidate_corners(candidates)
        # Get test and true values
        true_cand, true_cont = aruco._filterTooCloseCandidates(candidates,
                                                               contours,
                                                               MarkerDetectPar.params[MarkerDetectPar.minMarkerDistanceRate])
        test_cand, test_cont = MarkerDetectPar._filter_too_close_candidates(candidates, contours)
        # Assert equality
        np.testing.assert_allclose(test_cand, true_cand)
        np.testing.assert_array_equal(true_cand, true_cont)

    def test_filter_too_close_candidates_does_not_alter_params(self):
        candidates, contours = aruco._detectInitialCandidates(self.gray)
        cand_copy, cont_copy = np.copy(candidates), np.copy(contours)
        MarkerDetectPar._filter_too_close_candidates(candidates, contours)
        np.testing.assert_allclose(candidates, cand_copy)
        np.testing.assert_array_equal(contours, cont_copy)

    # ~~STEP 2 FUNCTIONS~~
    @unittest.skip("Faulty Python binding")
    def test_identify_candidates_equals_aruco_method(self):
        candidates, contours = aruco._detectCandidates(self.gray, aruco.DetectorParameters_create())
        aruco_acc, aruco_rej, aruco_ids = list(), list(), list()
        aruco_ids, aruco_rej = aruco._identifyCandidates(self.gray, candidates, contours,
                                                         AeroCubeMarker.get_dictionary(),
                                                         aruco_acc,
                                                         aruco.DetectorParameters_create())

    def test_extract_bits_equals_aruco_method(self):
        candidates, contours = aruco._detectCandidates(self.gray_marker, aruco.DetectorParameters_create())
        print(candidates[9])
        # true_bits = aruco._extractBits(self.gray_marker, candidates[9], FiducialMarker.get_marker_size(),
        #                                MarkerDetectPar.params[MarkerDetectPar.markerBorderBits],
        #                                MarkerDetectPar.params[MarkerDetectPar.perspectiveRemovePixelPerCell],
        #                                MarkerDetectPar.params[MarkerDetectPar.perspectiveRemoveIgnoredMarginPerCell],
        #                                MarkerDetectPar.params[MarkerDetectPar.minOtsuStdDev])
        test_bits = MarkerDetectPar._extract_bits(self.gray_marker, candidates[9])
        all_the_bits = [MarkerDetectPar._extract_bits(self.gray_marker, c) for c in candidates]
        print(test_bits)


    # ~~STEP 3 FUNCTIONS~~

    # ~~STEP 4 FUNCTIONS~~

if __name__ == '__main__':
    unittest.main()
