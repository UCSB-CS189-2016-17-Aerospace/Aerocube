import os
import unittest
import numpy as np
import cv2
import tempfile
from aerocubeMarker import AeroCubeMarker
from cameraCalibration import CameraCalibration


class TestCameraCalibration(unittest.TestCase):
    def test_predefined_calibration_andrew_iphone(self):
        cal = CameraCalibration.PredefinedCalibration.ANDREW_IPHONE
        self.assertEqual(cal._fields,
                         ('RET_VAL', 'CAMERA_MATRIX', 'DIST_COEFFS'))
        self.assertEqual(cal.CAMERA_MATRIX.shape, (3, 3))
        self.assertEqual(cal.DIST_COEFFS.shape, (1, 5))

    def test_predefined_calibration_nonexistent_calibration(self):
        with self.assertRaises(AttributeError):
            cal = CameraCalibration.PredefinedCalibration.TRASH_VALUE

    def test_get_charucoboard(self):
        board = CameraCalibration.get_charucoboard()
        self.assertEqual(board.getChessboardSize(), (8, 5))
        self.assertEqual(board.getMarkerLength(), 9)
        self.assertEqual(board.getSquareLength(), 10)
        self.assertTrue(AeroCubeMarker.dictionary_equal(
                                            board.dictionary,
                                            AeroCubeMarker.get_dictionary())
                        )

    def test_draw_charucoboard(self):
        fp = tempfile.NamedTemporaryFile(dir='test_files', suffix='.jpg')
        CameraCalibration.draw_charucoboard((80, 50), fp.name)
        self.assertTrue(os.stat(fp.name).st_size != 0)
        fp.close()

    def test_draw_charucoboard_invalid_out_size(self):
        fp = tempfile.NamedTemporaryFile(dir='test_files', suffix='.jpg')
        self.assertRaises(Exception,
                          CameraCalibration.draw_charucoboard,
                          (48, 50),
                          fp.name)
        fp.close()

    def test_get_calibration_matrices(self):
        board = CameraCalibration.get_charucoboard()
        img_paths = ["./test_files/andrew_iphone_calibration_photo_0.jpg",
                     "./test_files/andrew_iphone_calibration_photo_1.jpg",
                     "./test_files/andrew_iphone_calibration_photo_2.jpg",
                     "./test_files/andrew_iphone_calibration_photo_3.jpg"]
        img_arr = [cv2.imread(img) for img in img_paths]
        retval = CameraCalibration.get_calibration_matrices(board, img_arr)
        self.assertEqual(len(retval), 3)
        self.assertEqual(retval[1].shape, (3, 3))
        self.assertEqual(retval[2].shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
