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
import numpy as np

# img_path = "ImP/imageProcessing/test_files/jetson_test1.jpg"
# img_path = "ImP/imageProcessing/test_files/capstone_class_photoshoot/AC_0_125_DOCT.JPG"
img_path = "ImP/imageProcessing/test_files/capstone_class_photoshoot/SPACE_0.JPG"
imp = ImageProcessor(img_path)
corners, ids = imp._find_fiducial_markers()
rvecs, tvecs = imp._find_pose(corners)
quaternions = [imp.rodrigues_to_quaternion(r) for r in rvecs]
[print(q == q.normalised) for q in quaternions]
print(corners)
print(quaternions)
print(tvecs)

img = aruco.drawAxis(np.copy(imp._img_mat), imp._cal.CAMERA_MATRIX, imp._cal.DIST_COEFFS, rvecs[-1], tvecs[-1], ImageProcessingSettings.get_marker_length())
cv2.imwrite("ImP/output_files/OUTPUT_1.jpg", img)
img = imp.draw_aerocube_markers()
cv2.imwrite("ImP/output_files/OUTPUT.jpg", img)
img = imp.draw_aerocubes()
cv2.imwrite("ImP/output_files/OUTPUT_CUBE.jpg", img)
