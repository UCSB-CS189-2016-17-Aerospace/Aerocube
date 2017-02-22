import os
import unittest
import timeit
import cProfile
import pstats
import io
import numpy as np
import cv2
from cv2 import aruco
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
import ImP.imageProcessing.parallel.markerDetectPar as MarkerDetectPar
from ImP.imageProcessing.settings import ImageProcessingSettings
import ImP.imageProcessing.parallel.cuda.GpuWrapper as GpuWrapper
GpuWrapper.init()


class TestMarkerDetectParPerf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._CAPSTONE_PHOTO_PATH = os.path.join(ImageProcessingSettings.get_test_files_path(), 'capstone_class_photoshoot')
        cls._IMAGE_PATH = os.path.join(ImageProcessingSettings.get_test_files_path(), 'jetson_test1.jpg')
        cls._IMAGE = cv2.imread(cls._IMAGE_PATH)
        cls._IMG_MARKER_0_PATH = os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0.png')
        cls._IMG_MARKER_0 = cv2.imread(cls._IMG_MARKER_0_PATH)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.image = np.copy(self._IMAGE)
        self.img_marker_0 = np.copy(self._IMG_MARKER_0)
        self.gray_marker_0 = cv2.cvtColor(self.img_marker_0, cv2.COLOR_BGR2GRAY)

    def tearDown(self):
        pass

    # Test and profile entire algorithm

    def test_time_serial_vs_par(self):
        img_path = os.path.join(self._CAPSTONE_PHOTO_PATH, 'AC_0_1_FRONT_TOP.JPG')

        imp = ImageProcessor(img_path)
        pr = cProfile.Profile()
        pr.enable()
        actual_corners, actual_ids = imp._find_fiducial_markers()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        pr = cProfile.Profile()
        pr.enable()
        test_corners, test_ids = imp._find_fiducial_markers(parallel=True)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        np.testing.assert_allclose(actual_corners, test_corners)
        np.testing.assert_array_equal(actual_ids, test_ids)

    # Test GPU-related methods

    def test_cuda_cvt_color_gray(self):
        pr = cProfile.Profile()
        pr.enable()
        actual_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        pr = cProfile.Profile()
        pr.enable()
        test_gray = GpuWrapper.cudaCvtColorGray(self.image)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        np.testing.assert_allclose(actual_gray, test_gray)

    def test_cuda_warp_perspective_equals_warp_perspective(self):
        # self.fail()
        candidates, _ = aruco._detectCandidates(self.gray_marker_0, aruco.DetectorParameters_create())
        corners = candidates[9]
        src = self.gray_marker_0
        M = np.array([[ 7.03145227e-03,  5.50015822e-02, -5.41421824e+00],
                      [-6.65280496e-02, -4.24647125e-04,  2.78278338e+01],
                      [ 6.38002118e-04, -2.75228447e-04,  1.00000000e+00]], dtype=np.float32)
        result_img_size = 24
        result_img_corners = np.array([[ 0.,  0.],
                                       [23.,  0.],
                                       [23., 23.],
                                       [ 0., 23.]])

        pr = cProfile.Profile()
        pr.enable()
        actual_dst = cv2.warpPerspective(src, M, (result_img_size, result_img_size), flags=cv2.INTER_NEAREST)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        pr = cProfile.Profile()
        pr.enable()
        test_dst = GpuWrapper.cudaWarpPerspectiveWrapper(src.astype(dtype=np.uint8),
                                                         M.astype(dtype=np.float32),
                                                         (result_img_size, result_img_size),
                                                         _flags=cv2.INTER_NEAREST)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        pr = cProfile.Profile()
        pr.enable()
        test_host_dst = GpuWrapper.warpPerspectiveWrapper(src.astype(dtype=np.uint8),
                                                          M.astype(dtype=np.float32),
                                                          (result_img_size, result_img_size),
                                                          _flags=cv2.INTER_NEAREST)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        np.testing.assert_array_equal(actual_dst, test_host_dst)
        np.testing.assert_array_equal(actual_dst, test_dst)

    def test_echo_from_cython(self):
        M = np.array([[7.03145227e-03, 5.50015822e-02, -5.41421824e+00],
                      [-6.65280496e-02, -4.24647125e-04, 2.78278338e+01],
                      [6.38002118e-04, -2.75228447e-04, 1.00000000e+00]], dtype=np.float32)
        test_M = GpuWrapper.echoPyObject(M)
        np.testing.assert_allclose(M, test_M)

if __name__ == '__main__':
    unittest.main()
