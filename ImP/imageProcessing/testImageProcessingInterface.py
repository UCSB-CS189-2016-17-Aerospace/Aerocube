import os
import unittest
from collections import namedtuple

import cv2
import numpy as np
from cv2 import aruco

from jobs.aeroCubeSignal import ImageEventSignal
from .aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCube
from .imageProcessingInterface import ImageProcessor
from .settings import ImageProcessingSettings


class TestImageProcessingInterfaceMethods(unittest.TestCase):
    test_files_path = ImageProcessingSettings.get_test_files_path()
    test_output_path = os.path.join(test_files_path, 'output.png')
    # create named tuple to help organize test files and expected results of scan
    TestFile = namedtuple('TestFile', 'img_path \
                                       corners \
                                       IDs')
    # struct for image 'marker_4X4_sp6_id0.png'
    TEST_SINGLE_MARKER = TestFile(img_path=os.path.join(test_files_path, 'marker_4X4_sp6_id0.png'),
                                  corners=[np.array([[[82.,  51.],
                                                      [453., 51.],
                                                      [454., 417.],
                                                      [82.,  417.]]])],
                                  IDs=np.array([[0]]))
    # struct for image 'no_markers.jpg'
    TEST_NO_MARKER = TestFile(img_path=os.path.join(test_files_path, 'no_markers.jpg'),
                              corners=[],
                              IDs=None)
    # struct for image '2_ZENITH_0_BACK.jpg'
    TEST_MULT_AEROCUBES = TestFile(img_path=os.path.join(test_files_path, '2_ZENITH_0_BACK.jpg'),
                                   corners=[np.array([[[371.,  446.],
                                                       [396.,  505.],
                                                       [312.,  533.],
                                                       [290.,  471.]]]),
                                            np.array([[[801.,  314.],
                                                       [842.,  359.],
                                                       [779.,  380.],
                                                       [741.,  333.]]])],
                                   IDs=np.array([[12], [4]]))

    def test_init(self):
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        self.assertIsNotNone(imp._img_mat)

    def test_positive_load_image(self):
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        self.assertIsNotNone(ImageProcessor._load_image(self.TEST_SINGLE_MARKER.img_path))

    def test_negative_load_image(self):
        self.assertRaises(OSError, ImageProcessor, self.TEST_SINGLE_MARKER.img_path + "NULL")

    def test_find_fiducial_marker(self):
        # hard code results of operation
        corners = self.TEST_SINGLE_MARKER.corners
        ids = self.TEST_SINGLE_MARKER.IDs
        # get results of function
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        test_corners, test_ids = imp._find_fiducial_markers()
        # assert hard-coded results equal results of function
        self.assertTrue(np.array_equal(corners, test_corners))
        self.assertTrue(np.array_equal(ids, test_ids))
        self.assertEqual(type(test_ids).__module__, np.__name__)
        # save output image for visual confirmation
        output_img = aruco.drawDetectedMarkers(imp._img_mat, test_corners, test_ids)
        cv2.imwrite(self.test_output_path, output_img)

    def test_find_fiducial_marker_multiple(self):
        # get results from ImP
        imp = ImageProcessor(self.TEST_MULT_AEROCUBES.img_path)
        test_corners, test_ids = imp._find_fiducial_markers()
        # assert hard-coded results equal ImP results
        self.assertTrue(np.array_equal(self.TEST_MULT_AEROCUBES.corners, test_corners))
        self.assertTrue(np.array_equal(self.TEST_MULT_AEROCUBES.IDs, test_ids))

    def test_find_fiducial_marker_none(self):
        imp = ImageProcessor(self.TEST_NO_MARKER.img_path)
        test_corners, test_ids = imp._find_fiducial_markers()
        self.assertTrue(np.array_equal(self.TEST_NO_MARKER.corners, test_corners))
        self.assertTrue(np.array_equal(self.TEST_NO_MARKER.IDs, test_ids))

    def test_find_aerocube_marker(self):
        # hard code results of operation
        aerocube_ID = 0
        aerocube_face = AeroCubeFace.ZENITH
        corners = self.TEST_SINGLE_MARKER.corners
        true_markers = np.array([AeroCubeMarker(aerocube_ID,
                                                aerocube_face,
                                                corners[0])])
        # get results of function
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        aerocube_markers = imp._find_aerocube_markers()
        # assert equality of arrays
        self.assertTrue(np.array_equal(true_markers, aerocube_markers))

    def test_find_aerocube_markers_multiple(self):
        # get hard-coded results
        aerocube_markers = [AeroCubeMarker(2, AeroCubeFace.ZENITH, self.TEST_MULT_AEROCUBES.corners[0]),
                            AeroCubeMarker(0,   AeroCubeFace.BACK, self.TEST_MULT_AEROCUBES.corners[1])]
        # get results from ImP
        imp = ImageProcessor(self.TEST_MULT_AEROCUBES.img_path)
        test_aerocube_markers = imp._find_aerocube_markers()
        # assert equality
        self.assertTrue(np.array_equal(aerocube_markers, test_aerocube_markers))

    def test_find_aerocube_markers_none(self):
        # get hard-coded results
        aerocube_markers = []
        # get ImP results
        imp = ImageProcessor(self.TEST_NO_MARKER.img_path)
        test_markers = imp._find_aerocube_markers()
        self.assertTrue(np.array_equal(aerocube_markers, test_markers))

    def test_identify_aerocubes(self):
        aerocube_ID = 0
        aerocube_face = AeroCubeFace.ZENITH
        corners = self.TEST_SINGLE_MARKER.corners
        marker_list = [AeroCubeMarker(aerocube_ID, aerocube_face, corners[0])]
        aerocube_list = [AeroCube(marker_list)]
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        self.assertEqual(imp._identify_aerocubes(), aerocube_list)

    def test_identify_aerocubes_multiple(self):
        # get hard-coded results
        corners = self.TEST_MULT_AEROCUBES.corners
        aerocube_2_markers = [AeroCubeMarker(2, AeroCubeFace.ZENITH, corners[0])]
        aerocube_0_markers = [AeroCubeMarker(0,   AeroCubeFace.BACK, corners[1])]
        aerocubes = [AeroCube(aerocube_2_markers), AeroCube(aerocube_0_markers)]
        # get ImP results
        imp = ImageProcessor(self.TEST_MULT_AEROCUBES.img_path)
        test_aerocubes = imp._identify_aerocubes()
        # assert equality
        self.assertTrue(np.array_equal(aerocubes, test_aerocubes))

    def test_identify_aerocubes_none(self):
        imp = ImageProcessor(self.TEST_NO_MARKER.img_path)
        self.assertEqual([], imp._identify_aerocubes())

    def test_scan_image(self):
        # get hard-coded results
        corners = self.TEST_SINGLE_MARKER.corners
        marker_list = [AeroCubeMarker(0, AeroCubeFace.ZENITH, corners[0])]
        aerocube_list = [AeroCube(marker_list)]
        # get ImP results
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        scan_results = imp.scan_image(ImageEventSignal.IDENTIFY_AEROCUBES)
        # assert equality
        self.assertTrue(np.array_equal(aerocube_list, scan_results))

    def test_draw_fiducial_markers(self):
        imp = ImageProcessor(self.TEST_SINGLE_MARKER.img_path)
        corners, IDs = imp._find_fiducial_markers()
        img = imp.draw_fiducial_markers(corners, IDs)
        self.assertEqual(img.shape, imp._img_mat.shape)

if __name__ == '__main__':
    unittest.main()
