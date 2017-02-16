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
imp = ImageProcessor(img_path, cal=CameraCalibration.PredefinedCalibration.GUS_GOPRO)
corners, ids = imp._find_fiducial_markers()
rvecs, tvecs = imp._find_pose()
quaternions = [imp.rodrigues_to_quaternion(r) for r in rvecs]
print(quaternions)
print(tvecs)
