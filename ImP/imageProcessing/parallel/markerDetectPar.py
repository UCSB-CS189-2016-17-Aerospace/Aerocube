import os
import math
import cv2
from cv2 import aruco
import ctypes
import numpy as np
import numba
from numba import cuda as nb_cuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker
# Import and initialize PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Ctypes Wrappers for CUDA Libraries

_SO_DIR = os.path.dirname(__file__)
_MARKER_DETECT_PAR_GPU = ctypes.cdll.LoadLibrary(os.path.join(_SO_DIR, 'libMarkerDetectParGPU.so'))


# Initialize warpPerspective


class CV_SIZE(ctypes.Structure):
    _fields_ = [
        ('height', ctypes.c_float),
        ('width', ctypes.c_float)
    ]

def _initialize_warp_perspective():
    """
    cv::cuda::warpPerspective function signature
    * http://docs.opencv.org/trunk/db/d29/group__cudawarping.html#ga7a6cf95065536712de6b155f3440ccff
    :return:
    """
    _func = _MARKER_DETECT_PAR_GPU.warpPerspectiveWrapper
    _func.restype = ctypes.c_int32
    c_float_p = ctypes.POINTER(ctypes.c_float)
    _func.argtypes = [c_float_p,
                      c_float_p,
                      c_float_p,
                      ctypes.POINTER(CV_SIZE),
                      ctypes.c_int32]
    return _func
    pass

# Assign wrapped functions to private module variables
_func_warp_perspective = _initialize_warp_perspective()


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
    @nb_cuda.jit(device=True)
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

    @staticmethod
    def _cuda_warp_perspective(src, M, dsize, flags=cv2.INTER_NEAREST):
        # TODO: convert src and dst to GpuMat
        # Get the warpPerspective function
        warpPerspective = _func_warp_perspective
        # Convert src, M to ctype-friendly format
        c_float_p = ctypes.POINTER(ctypes.c_float)
        src_ptr = src.astype(np.float32).ctypes.data_as(c_float_p)
        M_ptr = M.astype(np.float32).ctypes.data_as(c_float_p)
        # Create dst as ctype-friendly format
        dst_ptr = np.zeros(src.shape, dtype=np.float32).ctypes.data_as(c_float_p)
        # Convert dsize to specified format
        cv_size = CV_SIZE(*dsize)
        warpPerspective(src_ptr, dst_ptr, M_ptr, cv_size, flags)
        return dst_ptr.astype(np.float32)

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
    @nb_cuda.jit
    def cuda_increment_by_one(an_array):
        # Thread id in a 1D block
        tx = nb_cuda.threadIdx.x
        # Block id in a 1D grid
        ty = nb_cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = nb_cuda.blockDim.x
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

        # ~~STEP 2~~: Identify marker candidates, filtering out candidates without properly set bits
        accepted, ids, rejected = cls._identify_candidates(gray_img, candidates, dictionary)

        # ~~STEP 3~~: Filter detected markers
        candidates, ids = cls._filter_detected_markers(candidates, ids)

        # ~~STEP 4~~: Do corner refinement (if necessary)
        # Params default to False, so this is going to stay un-implemented!

        return candidates, ids


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
                  cls.params[cls.adaptiveThreshWinSizeStep] + 1
        # Declare candidates and contours arrays
        candidates = list()
        contours = list()

        # Threshold at different scales
        for i in range(int(nScales)):
            scale = cls.params[cls.adaptiveThreshWinSizeMin] + i * cls.params[cls.adaptiveThreshWinSizeStep]
            cand, cont = cls._find_marker_contours(cls._threshold(gray, scale, cls.params[cls.adaptiveThreshConstant]))
            if len(cand) > 0:
                for j in range(len(cand)):
                    candidates.append(cand[j])
                    contours.append(cont[j])

        return np.squeeze(candidates), contours

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
        return candidates, contours_out

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
        too close to each other. Does not modify the original params.
        :param candidates: list of candidates, each represented by four Points, with values Point(x,y)
        :param contours: list of contours, with points stored as ints
        :return: candidates, contours filtered by conditions described in comments
        """
        minMarkerDistanceRate = cls.params[cls.minMarkerDistanceRate]
        assert minMarkerDistanceRate >= 0
        # Initialize array to keep track of markers to remove
        to_remove = [False]*len(candidates)
        # Compare mean square distance between corner points for each candidate (squared to avoid negatives)
        # If a candidate's corner is too close (has a mean square distance too low) to the other candidate's corners,
        # remove the smaller candidate of the pair
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                minimumPerimeter = float(min(len(contours[i]), len(contours[j])))
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
        # Add marker info to out arrays only if not marked for removal
        cand_out = [c for i, c in enumerate(candidates) if to_remove[i] is False]
        cont_out = [c for i, c in enumerate(contours) if to_remove[i] is False]
        return cand_out, cont_out

    # ~~STEP 2 FUNCTIONS~~

    @classmethod
    def _identify_candidates(cls, gray, candidates, dictionary):
        """
        Iterates through given array of candidates and extracts the "bits" of the candidate marker by Otsu thresholding.
        Rejects the candidate if bits are not properly set or identifiable by the dictionary. Additionally, reverses any
        rotation applied to each candidate's corner points before returning them from this function.
        :param gray: grayscale image containing the candidate corner points
        :param candidates: list of each candidate's corner points, with shape (4, 2) and type np.float32
        :param dictionary: dictionary used to analyze and identify each candidate marker
        :return: (accepted, ids, rejected)
            * accepted - list of accepted candidates, with the corner points rotated to reflect their proper order
                according to the dictionary
            * ids - list of ids of the respective candidates, identified by the dictionary
            * rejected - list of rejected candidates (e.g., due to too many erroneous border bits, improper data bits,
                rejection by the dictionary)
        """
        # Assert that image is not none and gray
        assert gray is not None
        assert gray.size is not 0 and len(gray.shape) == 2
        # Initialize variables
        accepted = list()
        ids = list()
        rejected = list()
        # Analyze each candidate
        for i in range(len(candidates)):
            valid, corners, cand_id = cls._identify_one_candidate(dictionary, gray, candidates[i])
            if valid:
                accepted.append(corners)
                ids.append(cand_id)
            else:
                rejected.append(corners)

        return accepted, ids, rejected

    @classmethod
    def _identify_one_candidate(cls, dictionary, gray, corners):
        """
        Given a grayscale image and the candidate corners (i.e., corner points), extract the bits of the candidate from
        the image if possible and use the dictionary to identify the candidate. If successful, reverse any rotation
        applied to the ordering of the corner points (to standardize the order of the corners) and return with the ID.
        Else, return False, and return original corners and invalid ID.
        :param dictionary: dictionary used to identify extracted bits from the corners in the image
        :param gray: grayscale image that contains the corner points of the candidate
        :param corners: corner points of the candidate with shape (4,2); recall that Points are stored as [x][y]
        :return: (valid_candidate, corners, id)
            * valid_candidate - indicates whether the given candidate is valid or not, due to:
                1. too many erroneous bits (white bits) in candidate marker border (assumed to be all black bits)
                2. dictionary fails to identify the candidate marker
            * corners - corners of the identified candidate, rotated if the dictionary detects a rotation occurred;
                else, if candidate identified invalid, original corners returned
            * id - id of the identified candidate; if invalid candidate, set to -1
        """
        markerBorderBits = cls.params[cls.markerBorderBits]
        assert len(corners) is 4
        assert gray is not None
        assert markerBorderBits > 0

        # Get bits, and ensure there are not too many erroneous bits
        candidate_bits = cls._extract_bits(gray, corners)
        max_errors_in_border = int(dictionary.markerSize * dictionary.markerSize * markerBorderBits)
        border_errors = cls._get_border_errors(candidate_bits, dictionary.markerSize, markerBorderBits)
        if border_errors > max_errors_in_border:
            return False, corners, -1

        # Take inner bits for marker identification with beautiful Python slicing (god damn!)
        inner_bits = candidate_bits[markerBorderBits:-markerBorderBits, markerBorderBits:-markerBorderBits]
        retval, cand_id, rotation = dictionary.identify(inner_bits, cls.params[cls.errorCorrectionRate])
        if retval is False:
            return False, corners, -1
        else:
            # Shift corner positions to correct rotation before returning
            return True, np.roll(corners, rotation), cand_id
        pass

    @classmethod
    def _extract_bits(cls, gray, corners):
        """
        Extract the bits encoding the ID of the marker given the image and the marker's corners.
        First finds the perspective transformation matrix from the marker's "original" coordinates relative to the
        given image, then uses the transformation matrix to transform the entire image such that the marker's
        perspective is removed. Then performs thresholding on the marker (if appropriate) and counts the pixels in each
        cell (spatial area of one bit) to determine if "1" or "0".
        :param gray: grayscale image with the marker in question; undergoes a perspective transformation such that the
            original marker's perspective is removed, and analysis can occur
        :param corners: corner points of the marker in the grayscale image; must be in correct order (clockwise), such
            that the mapping from the marker's original coordinates to the marker's grayscale image coordinates is
            calculated correctly
        :return: 2-dimensional array of binary values representing the marker; for a 4x4 marker with default detector
            params, bits would be (4 inner bits + 2 border bits)^2 = 36 bits
        """
        # Initialize variables
        markerSize = FiducialMarker.get_marker_size()  # size of inner region of marker (area containing ID information)
        markerBorderBits = cls.params[cls.markerBorderBits]  # size of marker border
        cellSize = cls.params[cls.perspectiveRemovePixelPerCell]  # size of "cell", area consisting of one bit of info.
        cellMarginRate = cls.params[cls.perspectiveRemoveIgnoredMarginPerCell]  # cell margin
        minStdDevOtsu = cls.params[cls.minOtsuStdDev]  # min. std. dev. needed to run Otsu thresholding

        # Run assertions
        assert len(gray.shape) == 2
        assert len(corners) == 4
        assert markerBorderBits > 0 and cellSize > 0 and cellMarginRate >= 0 and cellMarginRate <= 1
        assert minStdDevOtsu >= 0

        # Determine new dimensions of perspective-removed marker
        markerSizeWithBorders = markerSize + 2*markerBorderBits
        cellMarginPixels = int(cellMarginRate * cellSize)
        resultImgSize = int(markerSizeWithBorders * cellSize)
        # Initialize corner matrix of perspective-removed marker to calculate perspective transformation matrix
        resultImgCorners = np.array([[0                , 0                ],
                                     [resultImgSize - 1, 0                ],
                                     [resultImgSize - 1, resultImgSize - 1],
                                     [0                , resultImgSize - 1]], dtype=np.float32)

        # Get transformation and apply to original imageimage
        transformation = cv2.getPerspectiveTransform(corners, resultImgCorners)
        result_img = cv2.warpPerspective(gray, transformation, (resultImgSize, resultImgSize), flags=cv2.INTER_NEAREST)

        # Initialize matrix containing bits output
        bits = np.zeros((markerSizeWithBorders, markerSizeWithBorders), dtype=np.int8)

        # Remove some border to avoid noise from perspective transformation
        # Remember that image matrices are stored row-major-order, [y][x]
        inner_region = result_img[int(cellSize/2):int(-cellSize/2), int(cellSize/2):int(-cellSize/2)]

        # Check if standard deviation enough to apply Otsu thresholding
        # If not enough, probably means all bits are same color (black or white)
        mean, stddev = cv2.meanStdDev(inner_region)
        if stddev < minStdDevOtsu:
            bits.fill(1) if mean > 127 else bits
            return bits

        # Because standard deviation is high enough, threshold using Otsu
        _, result_img = cv2.threshold(result_img, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        for y in range(markerSizeWithBorders):
            for x in range(markerSizeWithBorders):
                # Get each individual square of each cell, excluding the margin pixels
                yStart = y * cellSize + cellMarginPixels
                yEnd = yStart + cellSize - 2 * cellMarginPixels
                xStart = x * cellSize + cellMarginPixels
                xEnd = xStart + cellSize - 2 * cellMarginPixels
                square = result_img[yStart:yEnd, xStart:xEnd]
                if cv2.countNonZero(square) > (square.size / 2):
                    bits[y][x] = 1
        return bits

    @classmethod
    def _get_border_errors(cls, bits, marker_size, border_size):
        """
        Return number of erroneous bits in border (i.e., number of white bits in border).
        :param bits: 2-dimensional matrix of binary values, representing the bits (incl. border) of a marker
        :param marker_size: size of the marker, in terms of bits
        :param border_size: size of the marker border, in terms of bits
        :return: total count of white bits found in border
        """
        size_with_borders = marker_size + 2 * border_size
        assert marker_size > 0 and bits.shape == (size_with_borders, size_with_borders)

        # Iterate through border bits, counting number of white bits
        # Remember that bits (as with all image matrices) are stored row-major-order, where img[y][x]
        total_errors = 0
        for y in range(size_with_borders):
            for k in range(border_size):
                if bits[y][k] != 0: total_errors += 1
                if bits[y][size_with_borders - 1 - k] != 0: total_errors += 1
        for x in range(border_size, size_with_borders - border_size):
            for k in range(border_size):
                if bits[k][x] != 0: total_errors += 1
                if bits[size_with_borders - 1 - k][x] != 0: total_errors += 1

        return total_errors

    # ~~STEP 3 FUNCTIONS~~
    @classmethod
    def _filter_detected_markers(cls, corners, ids):
        """
        Filter markers that share the same ID by
        :param corners:
        :param ids:
        :return: (corners, ids) tuple
        """
        # Check that corners size is equal to id size, not sure if assert is done correctly
        assert len(corners) == len(ids)

        # If no markers detected, return immediately
        if len(corners) == 0:
            return corners, ids

        # Mark markers that will be deleted, initializes array all set to false
        to_remove = np.array([False] * len(corners))

        # Remove repeated markers with same id, or if one contains the other (double border bug)
        for i in range(len(corners) - 1):
            for j in range(1, len(corners)):
                if ids[i] != ids[j]:
                    continue
                else:
                    # Remove one of two identical (same ID) markers
                    # Check if first marker is inside second
                    inside = True
                    for p in range(4):
                        point = tuple(corners[j][p])
                        if cv2.pointPolygonTest(corners[i], point, measureDist=False) < 0:
                            inside = False
                            break
                    if inside:
                        to_remove[j] = True
                        continue

                    # check the second marker
                    inside = True
                    for p in range(4):
                        point = tuple(corners[i][p])
                        if cv2.pointPolygonTest(corners[j], point, measureDist=False) < 0:
                            inside = False
                            break
                    if inside:
                        to_remove[i] = True
                        continue

        corners = np.array(corners)
        ids = np.array(ids)
        filtered_corners = corners[to_remove != True]
        filtered_ids = ids[to_remove != True]

        return filtered_corners, filtered_ids

    @classmethod
    def _copy_vector_to_output(cls, vec, out):
        pass

                # ~~STEP 4 FUNCTIONS~~
