"""
Helper script for testing purposes.
This file can be called from the python3 shell as such:
exec(open("ImP/startImP.py").read())
"""

from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.cameraCalibration import CameraCalibration
from ImP.imageProcessing.settings import ImageProcessingSettings
from cv2 import aruco

img_path = "ImP/imageProcessing/test_files/jetson_test1.jpg"
imp = ImageProcessor(img_path)
rvecs, tvecs = imp._find_pose()
img = imp.draw_axis(CameraCalibration.PredefinedCalibration.ANDREW_IPHONE.CAMERA_MATRIX,
                    CameraCalibration.PredefinedCalibration.ANDREW_IPHONE.DIST_COEFFS,
                    imp.rodrigues_to_quaternion(rvecs[0]), tvecs, ImageProcessingSettings.get_marker_length())

img_true = aruco.drawAxis(imp._img_mat,
               CameraCalibration.PredefinedCalibration.ANDREW_IPHONE.CAMERA_MATRIX,
               CameraCalibration.PredefinedCalibration.ANDREW_IPHONE.DIST_COEFFS,
               rvecs, tvecs,
               ImageProcessingSettings.get_marker_length())
