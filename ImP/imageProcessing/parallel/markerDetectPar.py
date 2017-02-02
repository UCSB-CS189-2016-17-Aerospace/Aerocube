import cv2
import numpy as np
import numba
from numba import cuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker
# Import and initialize PyCUDA
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule


class MarkerDetectPar:
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

    class MarkerDetectParException(Exception):
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
        assert winSize >= 3
        if winSize % 2 == 0:
            winSize += 1
        maxValue = 255
        return cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY_INV, blockSize=winSize, C=constant)

    @staticmethod
    @cuda.jit(device=True)
    def _cuda_threshold(gray, winSize, constant=detectorParams[adaptiveThreshConstant]):
        """
        CUDA-accelerated implementation of threshold; should match _threshold's output.
        :param gray:
        :param winSize:
        :param constant:
        :return:
        """
        assert winSize >= 3
        if winSize % 2 == 0:
            winSize += 1
        maxValue = 255
        pass

    # @classmethod
    # def cuda_hello_world_print(cls):
    #     mod = SourceModule("""
    #             __global__ void multiply_them(float *dest, float *a, float *b)
    #             {
    #                 const int i = threadIdx.x;
    #                 dest[i] = a[i] * b[i];
    #             }
    #         """)
    #     multiply_them = mod.get_function("multiply_them")
    #     a = np.random.randn(400).astype(np.float32)
    #     b = np.random.randn(400).astype(np.float32)
    #     dest = np.zeros_like(a)
    #     multiply_them(
    #         cuda.Out(dest), cuda.In(a), cuda.In(b),
    #         block=(400, 1, 1), grid=(1, 1)
    #     )
    #     print(dest - a * b)

    @staticmethod
    @numba.jit(nopython=True)
    def numba_jit_add(x, y):
        return x + y

    @staticmethod
    @cuda.jit
    def cuda_increment_by_one(an_array):
        # Thread id in a 1D block
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        ty = cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = cuda.blockDim.x
        # Compute flattened index inside the array
        pos = tx + ty * bw
        print(pos)
        if pos < an_array.size:  # Check array boundaries
            an_array[pos] += 1

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
            raise cls.MarkerDetectParException

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
            raise cls.MarkerDetectParException
        if gray.size is 0 or len(gray.shape) != 2:
            raise cls.MarkerDetectParException
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
        assert cls.detectorParams[cls.adaptiveThreshWinSizeMin] >= 3 and cls.detectorParams[cls.adaptiveThreshWinSizeMax] >= 3
        assert cls.detectorParams[cls.adaptiveThreshWinSizeMax] >= cls.detectorParams[cls.adaptiveThreshWinSizeMin]
        assert cls.detectorParams[cls.adaptiveThreshWinSizeStep] > 0

        # Determine number of window sizes, or scales, to apply thresholding
        nScales = (cls.detectorParams[cls.adaptiveThreshWinSizeMax] - cls.detectorParams[cls.adaptiveThreshWinSizeMin]) / \
                  cls.detectorParams[cls.adaptiveThreshWinSizeStep]

        # Run sanity check, and assert nScales is valid (non-zero)
        assert nScales > 0

        # In parallel, threshold at different scales
        pass

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
        assert (minPerimeterRate > 0 and maxPerimeterRate > 0 accuracyRate > 0 and
                minCornerDistanceRate >= 0 and minDistanceToBorder >= 0)

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
