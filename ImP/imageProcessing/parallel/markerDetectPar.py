import math
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
    params = {
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
    def _threshold(cls, gray, winSize, constant=params[adaptiveThreshConstant]):
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
    def _cuda_threshold(gray, winSize, constant=params[adaptiveThreshConstant]):
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
        assert img

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
        # 1. VERIFY IMAGE IS GRAY
        assert gray is not None
        assert gray.size is not 0 and len(gray.shape) == 2
        # 2. DETECT FIRST SET OF CANDIDATES
        candidates, contours = cls._detect_initial_candidates(gray)
        # 3. SORT CORNERS
        cls._reorder_candidate_corners(candidates)
        # 4. FILTER OUT NEAR CANDIDATE PAIRS
        return cls._filter_too_close_candidates(candidates, contours)

    @classmethod
    def _detect_initial_candidates(cls, gray):
        """
        Finds an initial set of potential markers (candidates) by thresholding the image at varying scales
        and determining if any marker objects are found.
        :param gray: grayscale image to be analyzed
        :return: marker candidates and marker contours
        """
        # Check if detection parameters are valid
        assert cls.params[cls.adaptiveThreshWinSizeMin] >= 3 and cls.params[cls.adaptiveThreshWinSizeMax] >= 3
        assert cls.params[cls.adaptiveThreshWinSizeMax] >= cls.params[cls.adaptiveThreshWinSizeMin]
        assert cls.params[cls.adaptiveThreshWinSizeStep] > 0

        # Initialize variables
        # Determine number of window sizes, or scales, to apply thresholding
        nScales = (cls.params[cls.adaptiveThreshWinSizeMax] - cls.params[cls.adaptiveThreshWinSizeMin]) / \
                  cls.params[cls.adaptiveThreshWinSizeStep]
        # Declare candidates and contours arrays
        candidates = list()
        contours = list()

        # Threshold at different scales
        for i in range(int(nScales)):
            scale = cls.params[cls.adaptiveThreshWinSizeMin] + i * cls.params[cls.adaptiveThreshWinSizeStep]
            markers = cls._find_marker_contours(cls._threshold(gray, scale, cls.params[cls.adaptiveThreshConstant]))
            if len(markers[0]) > 0:
                candidates.append(markers[0])
                contours.append(markers[1])

        return np.squeeze(candidates), np.squeeze(contours)

    @classmethod
    def _find_marker_contours(cls, thresh):
        """
        Given a thresholded image, find candidate marker contours.
        :param thresh: thresholded image to find markers in
        :return: (candidates, contours)
        """
        # Set parameters
        minPerimeterRate = cls.params[cls.minMarkerPerimeterRate]
        maxPerimeterRate = cls.params[cls.maxMarkerPerimeterRate]
        accuracyRate = cls.params[cls.polygonalApproxAccuracyRate]
        minCornerDistanceRate = cls.params[cls.minCornerDistanceRate]
        minDistanceToBorder = cls.params[cls.minDistanceToBorder]

        # Assert parameters are valid
        assert (minPerimeterRate > 0 and maxPerimeterRate > 0 and accuracyRate > 0 and
                minCornerDistanceRate >= 0 and minDistanceToBorder >= 0)

        # Calculate maximum and minimum sizes in pixels based off of dimensions of thresh image
        minPerimeterPixels = minPerimeterRate * max(thresh.shape)
        maxPerimeterPixels = maxPerimeterRate * max(thresh.shape)

        # Get contours
        # Supply a copy of thresh to findContours, as it modifies the source image
        # RETR_LIST returns contours without any hierarchical relationships (as list)
        # CHAIN_APPROX_NONE stores all contour points, s.t. subsequent points of a contour are no further than 1 unit
        #   away from each other
        contours_img = np.copy(thresh)
        contours_img, contours, _ = cv2.findContours(contours_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Initialize candidates and contours arrays
        candidates = list()
        contours_out = list()

        # Filter list of contours
        for c in contours:
            # Check perimeter
            if len(c) < minPerimeterPixels or len(c) > maxPerimeterPixels:
                continue
            # Check is square (4 corners) and convex
            # "Squeeze" (remove dimensions 1-long) from approxCurve for more sensible indexing
            approxCurve = np.squeeze(cv2.approxPolyDP(c, len(c) * accuracyRate, True))
            if len(approxCurve) != 4 or not cv2.isContourConvex(approxCurve):
                continue
            # Check min distance between corners
            # Note that the "points" of approxCurve are stored as Points[col, row] --> Points[x,y],
            # whereas contours_img is a Image[row,col] --> Image[y,x]
            minDistSq = math.pow(max(contours_img.shape), 2)
            for j in range(4):
                minDistSq = min(math.pow(approxCurve[j][0] - approxCurve[(j+1) % 4][0], 2) +
                                math.pow(approxCurve[j][1] - approxCurve[(j+1) % 4][1], 2),
                                minDistSq)
            if minDistSq < math.pow(len(c) * minCornerDistanceRate, 2): continue
            # Check if it's too near to the img border
            # Note that images are stored similarly to matrices (row-major-order)
            # http://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
            too_near_border = False
            for pt in approxCurve:
                if (pt[0] < minDistanceToBorder or pt[1] < minDistanceToBorder or
                        pt[0] > contours_img.shape[1] - 1 - minDistanceToBorder or
                        pt[1] > contours_img.shape[0] - 1 - minDistanceToBorder):
                    too_near_border = True
            if too_near_border: continue
            # If all tests pass, add to candidate vector
            candidates.append(np.array(approxCurve, dtype=np.float32))
            contours_out.append(np.squeeze(c))

        return np.array(candidates), contours_out

    @classmethod
    def _reorder_candidate_corners(cls, candidates):
        """
        Reorder candidate corners to assure clockwise direction. Alters the original candidates array.
        Returns a reference to the candidates array (for convenience).
        :param candidates: List of candidates, each candidate being a list of four points, with values Point(x,y)
        :return: candidates list with reordered points, if necessary
        """
        for c in candidates:
            # Take distance from pt1 to pt0
            dx1 = c[1][0] - c[0][0]
            dy1 = c[1][1] - c[0][1]
            # Take distance from pt2 to pt0
            dx2 = c[2][0] - c[0][0]
            dy2 = c[2][1] - c[0][1]
            cross_product = float(dx1*dy2 - dy1*dx2)
            # If cross_product is counter-clockwise, swap pt1 and pt3
            if cross_product < 0.0:
                c[1], c[3] = c[3], c[1]
        return candidates

    @classmethod
    def _filter_too_close_candidates(cls, candidates, contours):
        """
        Modifies the given arrays of markers, filtering out candidate markers that are
        too close to each other.
        :param candidates: list of candidates, each represented by four Points, with values Point(x,y)
        :param contours: ???
        :return: candidates, contours
        """
        minMarkerDistanceRate = cls.params[cls.minMarkerDistanceRate]
        assert minMarkerDistanceRate >= 0
        # Initialize array to keep track of markers to remove
        to_remove = [False]*len(candidates)
        # Compare mean square distance between corner points for each candidate (squared to avoid negatives)
        # If a candidate's corner is too close (has a mean square distance too low) to the other candidate's corners,
        # remove the smaller candidate of the pair
        for i in range(len(candidates)):
            for j in range(1, len(candidates)):
                minimumPerimeter = int(min(len(contours[i]), len(contours[j])))
                minMarkerDistancePixels = minimumPerimeter * minMarkerDistanceRate
                # Because the corners (guaranteed clockwise) of i can have 4 different combinations with the
                # corners of j, we must repeat this process 4 times
                for fc in range(4):
                    # For each corner in candidate i, compute mean square distance to candidate j
                    distSq = 0
                    for c in range(4):
                        modC = (fc + c) % 4
                        distSq += math.pow(candidates[i][modC][0] - candidates[j][c][0], 2) + \
                                  math.pow(candidates[i][modC][1] - candidates[j][c][1], 2)
                    distSq /= 4.0  # Take the mean distance squared
                    # If mean square distance too low, mark for deletion
                    if distSq < math.pow(minMarkerDistancePixels, 2):
                        # If one marker already marked for deletion, do nothing
                        if to_remove[i] or to_remove[j]:
                            break
                        # Else, mark one with smaller contour perimeter for deletion
                        elif len(contours[i]) > len(contours[j]):
                            to_remove[j] = True
                        else:
                            to_remove[i] = True
        # Remove markers from candidates and contours array if marked for deletion
        del_markers = np.where(np.any(to_remove is True))
        return np.delete(candidates, del_markers), np.delete(contours, del_markers)


    # ~~STEP 2 FUNCTIONS~~

    # ~~STEP 3 FUNCTIONS~~

    # ~~STEP 4 FUNCTIONS~~
