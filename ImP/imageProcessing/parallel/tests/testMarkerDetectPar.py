import unittest
import os
import cv2
from cv2 import aruco
import numpy as np
from ImP.imageProcessing.aerocubeMarker import AeroCubeMarker
from ImP.imageProcessing.parallel.markerDetectPar import *
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.settings import ImageProcessingSettings
# Import the GpuWrapper and immediately initialize it
import ImP.imageProcessing.parallel.cython.GpuWrapper as GpuWrapper
GpuWrapper.init()


class TestMarkerDetectPar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._IMAGE = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'jetson_test1.jpg'))
        cls._IMG_MARKER_0 = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0.png'))
        cls._IMG_MARKER_0_TRANS = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0_transformed.png'))
        cls._CAPSTONE_PHOTO_DIR = os.path.join(ImageProcessingSettings.get_test_files_path(), 'capstone_class_photoshoot')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.image = np.copy(self._IMAGE)
        self.img_marker_0 = np.copy(self._IMG_MARKER_0)
        self.img_marker_0_trans = np.copy(self._IMG_MARKER_0_TRANS)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray_marker_0 = cv2.cvtColor(self.img_marker_0, cv2.COLOR_BGR2GRAY)
        self.gray_marker_0_trans = cv2.cvtColor(self.img_marker_0_trans, cv2.COLOR_BGR2GRAY)

    def tearDown(self):
        pass

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

    @unittest.expectedFailure
    def test_otsu_equal_to_without_otsu_thresholding(self):
        """
        Thresholding w/ Otsu vs. thresholding w/out Otsu gives a mismatch of 4.438% on the test image. If this is an
        acceptable margin of error and does not affect the end results, the thresholding calls with Otsu thresholding
        could be replaced with CUDA-accelerated thresholding techniques that do not support Otsu.
        :return:
        """
        no_otsu_rv, no_otsu = cv2.threshold(self.gray, 125, 255, cv2.THRESH_BINARY)
        otsu_rv, otsu = cv2.threshold(self.gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # print(no_otsu_rv)
        # print(otsu_rv)
        np.testing.assert_allclose(no_otsu, otsu)

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

    def test_detect_markers_parallel_does_not_break(self):
        MarkerDetectPar.detect_markers_parallel(self._IMG_MARKER_0)

    def test_detect_markers_parallel_on_capstone_photos(self):
        img_paths = [os.path.join(self._CAPSTONE_PHOTO_DIR, f) for f in os.listdir(self._CAPSTONE_PHOTO_DIR) if os.path.isfile(os.path.join(self._CAPSTONE_PHOTO_DIR, f))]
        for img_path in img_paths:
            imp = ImageProcessor(img_path)
            actual_corners, actual_ids = imp._find_fiducial_markers(parallel=True)
            expected_corners, expected_ids = imp._find_fiducial_markers(parallel=False)
            np.testing.assert_allclose(actual_corners, expected_corners)
            np.testing.assert_array_equal(actual_ids, expected_ids)
            print("PASSED: {}".format(img_path))

    # ~~STEP 1 FUNCTIONS~~

    def test_detect_candidates_equals_aruco_method(self):
        candidates, contours = MarkerDetectPar._detect_candidates(self.gray)
        aruco_cand, aruco_cont = aruco._detectCandidates(self.gray, aruco.DetectorParameters_create())
        np.testing.assert_allclose(candidates, aruco_cand)
        try:
            np.testing.assert_array_equal(contours, aruco_cont)
        except AssertionError:
            print("_detect_candidates: arrays not equal -- try per-element / per-row comparison")
            np.testing.assert_array_equal([np.array_equal(a, b) for a, b in zip(contours, aruco_cont)], [True]*len(contours))

    def test_detect_candidates_equals_aruco_method_simple(self):
        candidates, contours = MarkerDetectPar._detect_candidates(self.gray_marker_0)
        aruco_cand, aruco_cont = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        np.testing.assert_allclose(candidates, aruco_cand)
        try:
            np.testing.assert_array_equal(contours, aruco_cont)
        except AssertionError:
            print("_detect_candidates: arrays not equal -- try per-element / per-row comparison")
            np.testing.assert_array_equal([np.array_equal(a, b) for a, b in zip(contours, aruco_cont)], [True]*len(contours))

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
        try:
            np.testing.assert_array_equal(test_contours_thresh_3[1], true_contours_thresh_3[1])
        except AssertionError:
            print("_find_marker_contours: arrays not equal -- try per-element / per-row comparison")
            np.testing.assert_array_equal([np.array_equal(a, b) for a, b in zip(test_contours_thresh_3[1], true_contours_thresh_3[1])],
                                          [True]*len(test_contours_thresh_3[1]))
        # thresh with winSize = 13
        thresh_13 = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                          13, params[MarkerDetectPar.adaptiveThreshConstant])
        test_contours_thresh_13 = MarkerDetectPar._find_marker_contours(thresh_13)
        true_contours_thresh_13 = aruco._findMarkerContours(thresh_13, *aruco_params)
        np.testing.assert_allclose(test_contours_thresh_13[0], true_contours_thresh_13[0])
        try:
            np.testing.assert_array_equal(test_contours_thresh_13[1], true_contours_thresh_13[1])
        except AssertionError:
            print("_find_marker_contours: arrays not equal -- try per-element / per-row comparison")
            np.testing.assert_array_equal([np.array_equal(a, b) for a, b in zip(test_contours_thresh_13[1], true_contours_thresh_13[1])],
                                          [True]*len(test_contours_thresh_13[1]))

    def test_assert_find_marker_contours_does_not_modify_thresh(self):
        params = MarkerDetectPar.params
        thresh = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       3, params[MarkerDetectPar.adaptiveThreshConstant])
        thresh_copy = np.copy(thresh)
        MarkerDetectPar._find_marker_contours(thresh)
        np.testing.assert_equal(thresh, thresh_copy)

    @unittest.skip("_reorderCandidatesCorners: Python binding inadequate")
    def test_reorder_candidate_corners_equals_aruco_method(self):
        candidates, _ = aruco._detectInitialCandidates(self.gray)
        aruco._reorderCandidatesCorners(candidates)
        np.testing.assert_array_equal(MarkerDetectPar._reorder_candidate_corners(candidates),
                                      aruco._reorderCandidatesCorners(candidates))

    def test_reorder_candidate_corners_preserves_shape_and_alters_param(self):
        candidates, _ = aruco._detectInitialCandidates(self.gray)
        cand_copy = np.copy(candidates)
        tmp = MarkerDetectPar._reorder_candidate_corners(candidates)
        np.testing.assert_allclose(tmp, candidates)
        self.assertEqual(np.array([np.array_equal(c[1], c[3]) for c in candidates]).sum(), 0)
        self.assertFalse(np.allclose(cand_copy, candidates))
        self.assertFalse(np.allclose(cand_copy, tmp))
        self.assertEqual(np.array(candidates).shape, cand_copy.shape)

    @unittest.skip("_filterTooCloseCandidates: faulty Python binding")
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

    @unittest.skip("_identifyCandidates: faulty Python binding")
    def test_identify_candidates_equals_aruco_method(self):
        candidates, contours = aruco._detectCandidates(self.gray, aruco.DetectorParameters_create())
        aruco_acc, aruco_rej, aruco_ids = list(), list(), list()
        aruco_ids, aruco_rej = aruco._identifyCandidates(self.gray, candidates, contours,
                                                         AeroCubeMarker.get_dictionary(),
                                                         aruco_acc,
                                                         aruco.DetectorParameters_create())

    def test_identify_candidates_for_marker_returns_valid_results(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        acc, ids, rej = MarkerDetectPar._identify_candidates(self.gray_marker_0, candidates, FiducialMarker.get_dictionary())
        np.testing.assert_allclose(acc, [np.array([[ 82.,  51.],
                                                   [453.,  51.],
                                                   [454., 417.],
                                                   [ 82., 417.]], dtype=np.float32)])
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_allclose(rej, [np.array([[270., 297.],
                                                   [325., 297.],
                                                   [325., 352.],
                                                   [270., 352.]], dtype=np.float32),
                                         np.array([[332., 236.],
                                                   [387., 236.],
                                                   [387., 291.],
                                                   [332., 291.]], dtype=np.float32),
                                         np.array([[271., 236.],
                                                   [326., 236.],
                                                   [326., 291.],
                                                   [271., 291.]], dtype=np.float32),
                                         np.array([[331., 175.],
                                                   [386., 175.],
                                                   [386., 230.],
                                                   [331., 230.]], dtype=np.float32),
                                         np.array([[210., 175.],
                                                   [265., 175.],
                                                   [265., 230.],
                                                   [210., 230.]], dtype=np.float32),
                                         np.array([[332., 116.],
                                                   [387., 116.],
                                                   [387., 171.],
                                                   [332., 171.]], dtype=np.float32),
                                         np.array([[271., 116.],
                                                   [326., 116.],
                                                   [326., 171.],
                                                   [271., 171.]], dtype=np.float32),
                                         np.array([[147., 116.],
                                                   [202., 116.],
                                                   [202., 171.],
                                                   [147., 171.]], dtype=np.float32),
                                         np.array([[270., 177.],
                                                   [325., 176.],
                                                   [326., 230.],
                                                   [271., 231.]], dtype=np.float32)])
        candidates, _ = aruco._detectCandidates(self.gray_marker_0_trans, aruco.DetectorParameters_create())
        acc, ids, rej = MarkerDetectPar._identify_candidates(self.gray_marker_0_trans, candidates,
                                                             FiducialMarker.get_dictionary())
        # TODO: unsure if rotation was properly done here
        np.testing.assert_allclose(acc, [np.array([[418.,  45.],
                                                   [415., 515.],
                                                   [ 94., 475.],
                                                   [ 66.,  90.]], dtype=np.float32)])
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(np.array(rej).shape, (9, 4, 2))

    @unittest.skip("_identify_one_candidate: faulty Python binding, C++ assertion fails for valid Python-side input")
    def test_identify_one_candidate_equals_aruco_method(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        true_cand = aruco._identifyOneCandidate(FiducialMarker.get_dictionary(), self.gray_marker_0, candidates[9])

    def test_identify_one_candidate_returns_proper_id(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        retval = MarkerDetectPar._identify_one_candidate(FiducialMarker.get_dictionary(), self.gray_marker_0, candidates[9])
        self.assertTrue(retval[0])
        self.assertEqual(retval[2], 0)
        retval = MarkerDetectPar._identify_one_candidate(FiducialMarker.get_dictionary(), self.gray_marker_0, candidates[0])
        self.assertFalse(retval[0])
        self.assertEqual(retval[2], -1)

    @unittest.skip("_extractBits: faulty Python binding")
    def test_extract_bits_equals_aruco_method(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        true_bits = aruco._extractBits(self.gray_marker_0, candidates[9], FiducialMarker.get_marker_size(),
                                       MarkerDetectPar.params[MarkerDetectPar.markerBorderBits],
                                       MarkerDetectPar.params[MarkerDetectPar.perspectiveRemovePixelPerCell],
                                       MarkerDetectPar.params[MarkerDetectPar.perspectiveRemoveIgnoredMarginPerCell],
                                       MarkerDetectPar.params[MarkerDetectPar.minOtsuStdDev])
        test_bits = MarkerDetectPar._extract_bits(self.gray_marker_0, candidates[9])
        np.testing.assert_array_equal(true_bits, test_bits)

    def test_extract_bits_for_proper_marker(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        test_bits = MarkerDetectPar._extract_bits(self.gray_marker_0, candidates[9])
        np.testing.assert_array_equal(test_bits, np.array([[0, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 1, 1, 0],
                                                           [0, 0, 1, 0, 1, 0],
                                                           [0, 0, 0, 1, 1, 0],
                                                           [0, 0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 0, 0]]))
        candidates, _ = aruco._detectCandidates(self.gray_marker_0_trans, aruco.DetectorParameters_create())
        test_bits = MarkerDetectPar._extract_bits(self.gray_marker_0_trans, candidates[9])
        np.testing.assert_array_equal(test_bits, np.array([[0, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 1, 1, 0],
                                                           [0, 0, 1, 0, 1, 0],
                                                           [0, 0, 0, 1, 1, 0],
                                                           [0, 0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 0, 0]]))

    def test_extract_bits_for_false_candidate_all_black(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        bits = MarkerDetectPar._extract_bits(self.gray_marker_0, candidates[0])
        np.testing.assert_array_equal(bits, np.ones((6, 6), dtype=np.int8))


    def test_get_border_errors_equals_aruco_method(self):
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        bits_marker_0 = MarkerDetectPar._extract_bits(self.gray_marker_0, candidates[9])
        true_err_count = aruco._getBorderErrors(bits_marker_0,
                                                FiducialMarker.get_marker_size(),
                                                MarkerDetectPar.params[MarkerDetectPar.markerBorderBits])
        test_err_count = MarkerDetectPar._get_border_errors(bits_marker_0,
                                                            FiducialMarker.get_marker_size(),
                                                            MarkerDetectPar.params[MarkerDetectPar.markerBorderBits])
        self.assertEqual(test_err_count, true_err_count)
        candidates, _ = aruco._detectCandidates(self.gray_marker_0_trans, aruco.DetectorParameters_create())
        bits_marker_0_trans = MarkerDetectPar._extract_bits(self.gray_marker_0_trans, candidates[9])
        true_err_count = aruco._getBorderErrors(bits_marker_0_trans,
                                                FiducialMarker.get_marker_size(),
                                                MarkerDetectPar.params[MarkerDetectPar.markerBorderBits])
        test_err_count = MarkerDetectPar._get_border_errors(bits_marker_0_trans,
                                                            FiducialMarker.get_marker_size(),
                                                            MarkerDetectPar.params[MarkerDetectPar.markerBorderBits])
        self.assertEqual(test_err_count, true_err_count)

    # ~~STEP 3 FUNCTIONS~~

    def test_filter_detected_markers(self):
        # No Python implementation -- test if sensible results
        # Create corners for testing; make sure it is of type float32, or OpenCV will get angry in cv2.pointPolygonTest
        corners = np.array([[[1., 1.], [1., 5.], [5., 5.], [5., 1.]], [[2., 2.], [2., 4.], [4., 4.], [4., 2.]]],
                           dtype=np.float32)
        ids_diff = np.array([1, 2])
        ids_same = np.array([1, 1])
        test_corners, test_ids = MarkerDetectPar._filter_detected_markers(corners, ids_diff)
        np.testing.assert_allclose(test_corners, corners)
        np.testing.assert_array_equal(test_ids, ids_diff)
        test_corners, test_ids = MarkerDetectPar._filter_detected_markers(corners, ids_same)
        np.testing.assert_allclose(test_corners, np.array([[[1., 1.], [1., 5.], [5., 5.], [5., 1.]]]))
        np.testing.assert_array_equal(test_ids, [1])

    # ~~STEP 4 FUNCTIONS~~

    # Non-existent cause we don't have to implement -- yeah!

if __name__ == '__main__':
    unittest.main()
