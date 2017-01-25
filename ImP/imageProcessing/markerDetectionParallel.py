import cv2
import pycuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker


"""
Need tests everywhere.
* Test behavior for grayscale vs. non-grayscale input
"""


def detect_markers_CUDA(img, dictionary=FiducialMarker.get_dictionary()):
    """

    :param img:
    :param dictionary:
    :return:
    """
    # Raise exception if image is empty
    if not img:
        raise CUDAFunctionException

    # Convert to grayscale (if necessary)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 1: Detect marker candidates
    candidates, contours = _detect_candidates(gray_img)

    # STEP 2

    # STEP 3

    # STEP 4


def _detect_candidates(gray):
    """

    :param gray: grayscale image to be analyzed
    :return: marker candidates and marker contours
    """
    # Check if grayscale image is empty or is not actually a grayscale image
    if not gray and len(gray.shape) != 2:
        raise CUDAFunctionException


    pass


class CUDAFunctionException(Exception):
    """
    General exception for errors in the usage of the functions in this file.
    """