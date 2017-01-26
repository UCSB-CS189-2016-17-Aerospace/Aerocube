import cv2
import pycuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker


"""
Need tests everywhere.
* Test behavior for grayscale vs. non-grayscale input
"""
class MarkerDetectionParallel:

    # Set constants for parameter dict
    adaptiveThreshWinSizeMin = 'adaptiveThreshWinSizeMin'
    adaptiveThreshWinSizeMax = 'adaptiveThreshWinSizeMax'
    adaptiveThreshWinSizeStep = 'adaptiveThreshWinSizeStep'
    adaptiveThreshConstant = 'adaptiveThreshConstant'
    minMarkerPerimeterRate = 'minMarkerPerimeterRate'
    maxMarkerPerimeterRate = 'maxMarkerPerimeterRate'
    polygonalApproxAccuracyRate = 'polygonalApproxAccuracyRate'
    minCornerDistanceRate = 'minCornerDistanceRate'
    minDistanceToBorder = 'minDistanceToBorder'
    minMarkerDistanceRate = 'minMarkerDistanceRate'
    doCornerRefinement = 'doCornerRefinement'
    cornerRefinementWinSize = 'cornerRefinementWinSize'
    cornerRefinementMaxIterations = 'cornerRefinementMaxIterations'
    cornerRefinementMinAccuracy = 'cornerRefinementMinAccuracy'
    markerBorderBits = 'markerBorderBits'
    perspectiveRemovePixelPerCell = 'perspectiveRemovePixelPerCell'
    perspectiveRemoveIgnoredMarginPerCell = 'perspectiveRemoveIgnoredMarginPerCell'
    maxErroneousBitsInBorderRate = 'maxErroneousBitsInBorderRate'
    minOtsuStdDev = 'minOtsuStdDev'
    errorCorrectionRate = 'errorCorrectionRate'

    # parameter dictionary/values
    detectorParameters = {
        adaptiveThreshWinSizeMin: 3,
        adaptiveThreshWinSizeMax: 23,
        adaptiveThreshWinSizeStep: 10,
        adaptiveThreshConstant: 7,
        minMarkerPerimeterRate: 0.03,
        maxMarkerPerimeterRate: 4.,
        polygonalApproxAccuracyRate: 0.03,
        minCornerDistanceRate: 0.05,
        minDistanceToBorder: 3,
        minMarkerDistanceRate: 0.05,
        doCornerRefinement: False,
        cornerRefinementWinSize: 5,
        cornerRefinementMaxIterations: 30,
        cornerRefinementMinAccuracy: 0.1,
        markerBorderBits: 1,
        perspectiveRemovePixelPerCell: 4,
        perspectiveRemoveIgnoredMarginPerCell: 0.13,
        maxErroneousBitsInBorderRate: 0.35,
        minOtsuStdDev: 5.0,
        errorCorrectionRate: 0.6
    }

    @classmethod
    def detect_markers_CUDA(cls, img, dictionary=FiducialMarker.get_dictionary()):
        """

        :param img:
        :param dictionary:
        :return:
        """
        # Raise exception if image is empty
        if not img:
            raise cls.CUDAFunctionException

        # Convert to grayscale (if necessary)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # STEP 1: Detect marker candidates
        candidates, contours = cls.detect_candidates(gray_img)

        # STEP 2

        # STEP 3

        # STEP 4


    @classmethod
    def detect_candidates(cls, gray):
        """

        :param gray: grayscale image to be analyzed
        :return: marker candidates and marker contours
        """
        # Check if grayscale image is empty or is not actually a grayscale image
        if gray is None:
            raise cls.CUDAFunctionException
        if gray.size is 0 or len(gray.shape) != 2:
            raise cls.CUDAFunctionException
        pass




    @classmethod
    def detectInitialCandidates(cls, gray):
        """

        :param gray: grayscale image to be analyzed
        :return: marker candidates and marker contours
        """

        # Check if detection parameters are valid
        if cls.detectorParameters[cls.adaptiveThreshWinSizeMin] < 3 or cls.detectorParameters[cls.adaptiveThreshWinSizeMax] < 3:
            raise cls.CUDAFunctionException
        if cls.detectorParameters[cls.adaptiveThreshWinSizeMax] < cls.detectorParameters[cls.adaptiveThreshWinSizeMin]:
            raise cls.CUDAFunctionException
        if cls.detectorParameters[cls.adaptiveThreshWinSizeStep] <= 0:
            raise cls.CUDAFunctionException

        nScales = cls.detectorParameters[cls.adaptiveThreshWinSizeMax] - cls.detectorParameters[cls.adaptiveThreshWinSizeMin]

    class CUDAFunctionException(Exception):
        """
        General exception for errors in the usage of the functions in this file.
        """