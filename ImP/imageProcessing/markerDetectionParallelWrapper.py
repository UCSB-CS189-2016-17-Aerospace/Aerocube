import cv2
import pycuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker


class MarkerDetectionParallelWrapper:
    """
    Class wrapping together the logic of marker detection, using GPU parallelization when possible.

    TODO: Need tests everywhere.
    * Test behavior for grayscale vs. non-grayscale input
    """

    # ALGORITHM PARAMETERS

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

    # Parameter dictionary/values
    detectorParams = {
        adaptiveThreshWinSizeMin:               3,
        adaptiveThreshWinSizeMax:               23,
        adaptiveThreshWinSizeStep:              10,
        adaptiveThreshConstant:                 7,
        minMarkerPerimeterRate:                 0.03,
        maxMarkerPerimeterRate:                 4.,
        polygonalApproxAccuracyRate:            0.03,
        minCornerDistanceRate:                  0.05,
        minDistanceToBorder:                    3,
        minMarkerDistanceRate:                  0.05,
        doCornerRefinement:                     False,
        cornerRefinementWinSize:                5,
        cornerRefinementMaxIterations:          30,
        cornerRefinementMinAccuracy:            0.1,
        markerBorderBits:                       1,
        perspectiveRemovePixelPerCell:          4,
        perspectiveRemoveIgnoredMarginPerCell:  0.13,
        maxErroneousBitsInBorderRate:           0.35,
        minOtsuStdDev:                          5.0,
        errorCorrectionRate:                    0.6
    }

    # HELPER FUNCTIONS/OBJECTS

    class MarkerDetectionParallelException(Exception):
        """
        General exception for errors in the usage of the functions in this file.
        """

    @classmethod
    def _threshold(cls, gray, winSize, constant=detectorParams[adaptiveThreshConstant]):
        """
        Calls OpenCV's adaptiveThreshold method on what should be a grayscale image.
        Thresholds by looking at a block of pixels about a pixel (determined by winSize)
        and (because of ADAPTIVE_THRESH_MEAN_C) taking the mean of the pixel neighborhood.
        Thresholding is THRESH_BINARY_INV, meaning items that pass the threshold are set to 0; else, set to 1.
        Constant is subtracted from ADAPTIVE_THRESH_MEAN_C calculation to weight the thresholding.
        Reference:
        * http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold
        :param gray: image to be thresholded; must be single-channel (i.e., grayscale) image
        :param winSize: size of pixel neighborhood about a pixel, and therefore must be odd (e.g., 3, 5, 7, etc.)
        :param constant: used to weight the mean of the threshold calculations of a given pixel and its neighborhood
        :return: thresholded image
        """
        if winSize < 3:
            raise MarkerDetectionParallelWrapper.MarkerDetectionParallelException
        if winSize % 2 == 0:
            winSize += 1
        maxValue = 255
        return cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY_INV, blockSize=winSize, C=constant)


    # PUBLIC FUNCTIONS

    @classmethod
    def detect_markers_parallel(cls, img, dictionary=FiducialMarker.get_dictionary()):
        """
        Public entry point to algorithm. Delegates the steps of the algorithms to several helper functions.
        :param img: image that might contain markers; should not be a grayscale image
        :param dictionary: Aruco dictionary to identify markers from; defaults to the dictionary set in FiducialMarker
        :return:
        """
        # Raise exception if image is empty
        if not img:
            raise cls.MarkerDetectionParallelException

        # Convert to grayscale (if necessary)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ~~STEP 1~~: Detect marker candidates
        candidates, contours = cls._detect_candidates(gray_img)

        # STEP 2

        # STEP 3

        # STEP 4

    # ~~STEP 1 FUNCTIONS~~

    @classmethod
    def _detect_candidates(cls, gray):
        """
        Finds the initial candidates and contours of a grayscale image.
        :param gray: grayscale image to be analyzed
        :return: marker candidates and marker contours
        """
        # Check if grayscale image is empty or is not actually a grayscale image
        if gray is None:
            raise cls.MarkerDetectionParallelException
        if gray.size is 0 or len(gray.shape) != 2:
            raise cls.MarkerDetectionParallelException
        pass




    @classmethod
    def _detect_initial_candidates(cls, gray):
        """
        Finds an initial set of potential markers (candidates) by thresholding the image at varying scales
        and determining if any marker objects are found.
        :param gray: grayscale image to be analyzed
        :return: marker candidates and marker contours
        """

        # Check if detection parameters are valid
        if cls.detectorParams[cls.adaptiveThreshWinSizeMin] < 3 or cls.detectorParams[cls.adaptiveThreshWinSizeMax] < 3:
            raise cls.MarkerDetectionParallelException
        if cls.detectorParams[cls.adaptiveThreshWinSizeMax] < cls.detectorParams[cls.adaptiveThreshWinSizeMin]:
            raise cls.MarkerDetectionParallelException
        if cls.detectorParams[cls.adaptiveThreshWinSizeStep] <= 0:
            raise cls.MarkerDetectionParallelException

        # Determine number of window sizes, or scales, to apply thresholding
        nScales = (cls.detectorParams[cls.adaptiveThreshWinSizeMax] - cls.detectorParams[cls.adaptiveThreshWinSizeMin]) / \
                  cls.detectorParams[cls.adaptiveThreshWinSizeStep]

        # Run sanity check, and verify nScales is valid (non-zero)
        raise cls.MarkerDetectionParallelException if nScales <= 0 else None

    @classmethod
    def _find_marker_contours(cls, thresh):
        """
        Given a thresholded image, find candidate marker contours.
        :param thresh: thresholded image to find markers in
        :return: (candidates, contours)
        """
        # Set parameters
        minPerimeterRate = cls.detectorParams[cls.minMarkerPerimeterRate]
        maxPerimeterRate = cls.detectorParams[cls.maxMarkerPerimeterRate]
        accuracyRate = cls.detectorParams[cls.polygonalApproxAccuracyRate]
        minCornerDistanceRate = cls.detectorParams[cls.minCornerDistanceRate]
        minDistanceToBorder = cls.detectorParams[cls.minDistanceToBorder]

        # Assert parameters are valid
        if (minPerimeterRate <= 0 or maxPerimeterRate <= 0 or accuracyRate <= 0 or
            minCornerDistanceRate < 0 or minDistanceToBorder < 0):
            raise cls.MarkerDetectionParallelException

        # Calculate maximum and minimum sizes in pixels based off of dimensions of thresh image
        minPerimeterPixels = minPerimeterRate * max(thresh.shape)
        maxPerimeterPixels = maxPerimeterRate * max(thresh.shape)

        # Get contours
        # Supply a copy of thresh to findContours, as it modifies the source image
        # RETR_LIST returns contours without any hierarchical relationships (as list)
        # CHAIN_APPROX_NONE stores all contour points, s.t. subsequent points of a contour are no further than 1 unit
        #   away from each other
        contours_img = thresh
        contours_img, contours, _ = cv2.findContours(contours_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Filter list of contours
        pass


    # ~~STEP 2 FUNCTIONS~~

    # ~~STEP 3 FUNCTIONS~~

    # ~~STEP 4 FUNCTIONS~~