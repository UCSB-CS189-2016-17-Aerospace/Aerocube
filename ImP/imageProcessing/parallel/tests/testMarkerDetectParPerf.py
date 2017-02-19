import os
import unittest
import timeit
import cProfile
import pstats
import io
import numpy as np
import cv2
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.parallel.markerDetectPar import MarkerDetectPar
from ImP.imageProcessing.settings import ImageProcessingSettings
import ImP.imageProcessing.parallel.GpuWrapper as GpuWrapper


class TestMarkerDetectParPerf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._IMG_MARKER_0_PATH = os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0.png')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_time_serial_vs_par(self):
        imp = ImageProcessor(self._IMG_MARKER_0_PATH)
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

    def test_warp_perspective(self):
        GpuWrapper.cudaWarpPerspectiveWrapper()

if __name__ == '__main__':
    unittest.main()
