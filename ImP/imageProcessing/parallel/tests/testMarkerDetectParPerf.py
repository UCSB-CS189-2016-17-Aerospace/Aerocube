import os
import unittest
import timeit
import cProfile
import cv2
from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.parallel.markerDetectPar import MarkerDetectPar
from ImP.imageProcessing.settings import ImageProcessingSettings


class TestMarkerDetectParPerf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._IMG_MARKER_0 = cv2.imread(os.path.join(ImageProcessingSettings.get_test_files_path(), 'marker_4X4_sp6_id0.png'))

    @classmethod
    def tearDownClass(cls):
        pass

    def test_time_serial_vs_par(self):
        imp = ImageProcessor(self._IMG_MARKER_0)
        timeit.timeit('imp._find_fiducial_markers()')
        timeit.timeit('imp._find_fiducial_markers(parallel=True)')

if __name__ == '__main__':
    unittest.main()
