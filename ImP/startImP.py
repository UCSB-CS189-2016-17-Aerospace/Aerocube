"""
Helper script for testing purposes.
This file can be called from the python3 shell as such:
exec(open("ImP/startImP.py").read())
"""

from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.imageProcessing.cameraCalibration import CameraCalibration
from ImP.imageProcessing.settings import ImageProcessingSettings
import cv2
from cv2 import aruco

img_path = "ImP/imageProcessing/test_files/GOPR0040.JPG"
cal = CameraCalibration.PredefinedCalibration.GUS_GOPRO
imp = ImageProcessor(img_path)
corners, ids = imp._find_fiducial_markers()
rvecs, tvecs = imp._find_pose()
quaternions = [imp.rodrigues_to_quaternion(r) for r in rvecs]
print(quaternions)
print(tvecs)
img = imp.draw_fiducial_markers(corners, ids)
imp._img_mat = img
img1 = imp.draw_axis(cal.CAMERA_MATRIX, cal.DIST_COEFFS, quaternions[0], tvecs[0])
cv2.imwrite("ImP/output_files/GOPR0040_results_with_axes_img1.JPG", img1)
imp._img_mat = img1
img2 = imp.draw_axis(cal.CAMERA_MATRIX, cal.DIST_COEFFS, quaternions[1], tvecs[1])
cv2.imwrite("ImP/output_files/GOPR0040_results_with_axes_img2.JPG", img2)
