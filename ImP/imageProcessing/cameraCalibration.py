import cv2
from cv2 import aruco
import numpy as np
import os
from collections import namedtuple
from ImP.imageProcessing.aerocubeMarker import AeroCubeMarker
from ImP.imageProcessing.settings import ImageProcessingSettings


class CameraCalibration():
    """
    Manages camera calibration matrix and distortion coefficients.
    Example for calibrating with Python bindings for Charuco:
    http://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
    """
    class PredefinedCalibration():
        """
        Inner class to hold different calibration configurations constructed
        with named tuples. Configurations should be instances of _Calibration
        with constant-style names (e.g., all upper-case).
        """
        _Calibration = namedtuple('_Calibration', 'RET_VAL \
                                                   CAMERA_MATRIX \
                                                   DIST_COEFFS')
        ANDREW_IPHONE = _Calibration(
            RET_VAL=3.551523274640683,
            CAMERA_MATRIX=np.array([[3.48275636e+03, 0.00000000e+00, 2.02069885e+03],
                                    [0.00000000e+00, 3.52274282e+03, 1.51346685e+03],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            DIST_COEFFS=np.array([[-4.58647345e-02, 1.73122392e+00, -3.30440816e-03, -7.78486275e-04, -7.00795983e+00]])
        )

    @staticmethod
    def get_charucoboard():
        """
        Create a Charuco (chessboard Aruco) board using pre-set params.
        :return board: board object created by the aruco call
        """
        SQUARES_X = 8
        SQUARES_Y = 5
        SQUARE_LENGTH = 10
        MARKER_LENGTH = 9
        board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y,
                                          SQUARE_LENGTH,
                                          MARKER_LENGTH,
                                          AeroCubeMarker.get_dictionary())
        return board

    @staticmethod
    def draw_charucoboard(out_size, file_path):
        """
        Draw the Charucoboard to file_path
        :param out_size: size must be tuple (x, y) such that:
            x >= SQUARES_X * SQUARE_LENGTH
            y >= SQUARES_Y * SQUARE_LENGTH
        :param file_path: path to which the board image will be drawn
        """
        board = CameraCalibration.get_charucoboard()
        cv2.imwrite(file_path, aruco.drawPlanarBoard(board, out_size))

    @staticmethod
    def get_calibration_matrices(board, img_arr):
        """
        :param board: Charucoboard object to calibrate against
        :param img_arr: array of images (from different viewpoints)
        :return ret_val: unknown usage
        :return camera_matrix: 3X3 camera calibration matrix
        :return dist_coeffs: camera's distortion coefficients
        """
        dictionary = AeroCubeMarker.get_dictionary()
        all_charuco_corners = []
        all_charuco_IDs = []
        img_size = None
        for img in img_arr:
            # Convert to grayscale before performing operations
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, IDs, _ = aruco.detectMarkers(gray, dictionary)
            _, charuco_corners, charuco_IDs = aruco.interpolateCornersCharuco(
                                                                corners,
                                                                IDs,
                                                                gray,
                                                                board)
            all_charuco_corners.append(charuco_corners)
            all_charuco_IDs.append(charuco_IDs)
            # Get matrix shape of grayscale image
            img_size = gray.shape
        ret_val, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(all_charuco_corners,
                                                                                 all_charuco_IDs,
                                                                                 board,
                                                                                 img_size,
                                                                                 None,
                                                                                 None)
        return ret_val, camera_matrix, dist_coeffs

if __name__ == '__main__':
    # Get the calibration matrices for ANDREW_IPHONE calibration/configuration
    board = CameraCalibration.get_charucoboard()
    test_files_path = ImageProcessingSettings.get_test_files_path()
    img_paths = [os.path.join(test_files_path, "andrew_iphone_calibration_photo_0.jpg"),
                 os.path.join(test_files_path, "andrew_iphone_calibration_photo_1.jpg"),
                 os.path.join(test_files_path, "andrew_iphone_calibration_photo_2.jpg"),
                 os.path.join(test_files_path, "andrew_iphone_calibration_photo_3.jpg")]

    img_paths2 = [os.path.join(test_files_path, "gus_gorpo_1.jpg"),
                 os.path.join(test_files_path, "gus_gorpo_2.jpg"),
                 os.path.join(test_files_path, "gus_gorpo_3.jpg"),
                 os.path.join(test_files_path, "gus_gorpo_4.jpg")]

    img_arr = [cv2.imread(img) for img in img_paths2]
    print(CameraCalibration.get_calibration_matrices(board, img_arr))
