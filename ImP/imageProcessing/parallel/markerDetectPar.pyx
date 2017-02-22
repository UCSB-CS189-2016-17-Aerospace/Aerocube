import os
import math
import cv2
from cv2 import aruco
import numpy as np
cimport numpy as np
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker


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


def _threshold(gray, winSize, constant=params[adaptiveThreshConstant]):
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

# PUBLIC FUNCTIONS


def detect_markers_parallel(np.ndarray[dtype=np.uint8_t, ndim=3] img, dictionary=FiducialMarker.get_dictionary()):
    """
    Public entry point to algorithm. Delegates the steps of the algorithms to several helper functions.
    :param img: image that might contain markers; should not be a grayscale image
    :param dictionary: Aruco dictionary to identify markers from; defaults to the dictionary set in FiducialMarker
    :return:
    """
    # Raise exception if image is empty
    assert img is not None

    # Convert to grayscale (if necessary)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ~~STEP 1~~: Detect marker candidates
    candidates, contours = _detect_candidates(gray_img)

    # ~~STEP 2~~: Identify marker candidates, filtering out candidates without properly set bits
    accepted, ids, rejected = _identify_candidates(gray_img, candidates, dictionary)

    # ~~STEP 3~~: Filter detected markers
    filtered_candidates, ids = _filter_detected_markers(accepted, ids)

    # ~~STEP 4~~: Do corner refinement (if necessary)
    # Params default to False, so this is going to stay un-implemented!

    return filtered_candidates, ids


# ~~STEP 1 FUNCTIONS~~


def _detect_candidates(gray):
    """
    Finds the initial candidates and contours of a grayscale image.
    :param gray: grayscale image to be analyzed
    :return: marker candidates and marker contours
    """
    # 1. VERIFY IMAGE IS GRAY
    assert gray is not None
    assert gray.size is not 0 and len(gray.shape) == 2
    # 2. DETECT FIRST SET OF CANDIDATES
    candidates, contours = _detect_initial_candidates(gray)
    # 3. SORT CORNERS
    _reorder_candidate_corners(candidates)
    # 4. FILTER OUT NEAR CANDIDATE PAIRS
    return _filter_too_close_candidates(candidates, contours)


def _detect_initial_candidates(gray):
    """
    Finds an initial set of potential markers (candidates) by thresholding the image at varying scales
    and determining if any marker objects are found.
    :param gray: grayscale image to be analyzed
    :return: marker candidates and marker contours
    """
    # Check if detection parameters are valid
    assert params[adaptiveThreshWinSizeMin] >= 3 and params[adaptiveThreshWinSizeMax] >= 3
    assert params[adaptiveThreshWinSizeMax] >= params[adaptiveThreshWinSizeMin]
    assert params[adaptiveThreshWinSizeStep] > 0

    # Initialize variables
    # Determine number of window sizes, or scales, to apply thresholding
    nScales = (params[adaptiveThreshWinSizeMax] - params[adaptiveThreshWinSizeMin]) / \
              params[adaptiveThreshWinSizeStep] + 1
    # Declare candidates and contours arrays
    candidates = list()
    contours = list()

    # Threshold at different scales
    for i in range(int(nScales)):
        scale = params[adaptiveThreshWinSizeMin] + i * params[adaptiveThreshWinSizeStep]
        cand, cont = _find_marker_contours(_threshold(gray, scale, params[adaptiveThreshConstant]))
        if len(cand) > 0:
            for j in range(len(cand)):
                candidates.append(cand[j])
                contours.append(cont[j])

    return np.squeeze(candidates), contours


def _find_marker_contours(thresh):
    """
    Given a thresholded image, find candidate marker contours.
    :param thresh: thresholded image to find markers in
    :return: (candidates, contours)
    """
    # Set parameters
    min_perimeter_rate = params[minMarkerPerimeterRate]
    max_perimeter_rate = params[maxMarkerPerimeterRate]
    accuracy_rate = params[polygonalApproxAccuracyRate]
    min_corner_distance_rate = params[minCornerDistanceRate]
    min_distance_to_border = params[minDistanceToBorder]

    # Assert parameters are valid
    assert (min_perimeter_rate > 0 and max_perimeter_rate > 0 and accuracy_rate > 0 and
            min_corner_distance_rate >= 0 and min_distance_to_border >= 0)

    # Calculate maximum and minimum sizes in pixels based off of dimensions of thresh image
    minPerimeterPixels = min_perimeter_rate * max(thresh.shape)
    maxPerimeterPixels = max_perimeter_rate * max(thresh.shape)

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
        approxCurve = np.squeeze(cv2.approxPolyDP(c, len(c) * accuracy_rate, True))
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
        if minDistSq < math.pow(len(c) * min_corner_distance_rate, 2): continue
        # Check if it's too near to the img border
        # Note that images are stored similarly to matrices (row-major-order)
        # http://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
        too_near_border = False
        for pt in approxCurve:
            if (pt[0] < min_distance_to_border or pt[1] < min_distance_to_border or
                    pt[0] > contours_img.shape[1] - 1 - min_distance_to_border or
                    pt[1] > contours_img.shape[0] - 1 - min_distance_to_border):
                too_near_border = True
        if too_near_border: continue
        # If all tests pass, add to candidate vector
        candidates.append(np.array(approxCurve, dtype=np.float32))
        contours_out.append(np.squeeze(c))
    return candidates, contours_out


cdef void _reorder_candidate_corners(np.ndarray[dtype=np.float32_t, ndim=3] candidates):
    """
    Reorder candidate corners to assure clockwise direction. Alters the original candidates array.
    Returns a reference to the candidates array (for convenience).
    :param candidates: List of candidates, each candidate being a list of four points, with values Point(x,y)
    :return: nothing
    """
    cdef float dx1
    cdef float dy1
    cdef float dx2
    cdef float dy2
    cdef float cross_product
    cdef np.ndarray[dtype=np.float32_t, ndim=1] tmp
    for c in candidates:
        # Take distance from pt1 to pt0
        dx1 = c[1][0] - c[0][0]
        dy1 = c[1][1] - c[0][1]
        # Take distance from pt2 to pt0
        dx2 = c[2][0] - c[0][0]
        dy2 = c[2][1] - c[0][1]
        cross_product = dx1*dy2 - dy1*dx2
        # If cross_product is counter-clockwise, swap pt1 and pt3
        if cross_product < 0.0:
            # Swap elements -- note that data must be copied over, as otherwise tmp will just
            # be a reference pointing to c[3] (acting as another view into the data) instead of truly holding
            # c[3]'s original value
            # c[1], c[3] = c[3], c[1] does not work
            tmp = np.copy(c[3])
            c[3] = c[1]
            c[1] = tmp


def _filter_too_close_candidates(candidates, contours):
    """
    Modifies the given arrays of markers, filtering out candidate markers that are
    too close to each other. Does not modify the original params.
    :param candidates: list of candidates, each represented by four Points, with values Point(x,y)
    :param contours: list of contours, with points stored as ints
    :return: candidates, contours filtered by conditions described in comments
    """
    min_marker_distance_rate = params[minMarkerDistanceRate]
    assert min_marker_distance_rate >= 0
    # Initialize array to keep track of markers to remove
    to_remove = [False]*len(candidates)
    # Compare mean square distance between corner points for each candidate (squared to avoid negatives)
    # If a candidate's corner is too close (has a mean square distance too low) to the other candidate's corners,
    # remove the smaller candidate of the pair
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            minimumPerimeter = float(min(len(contours[i]), len(contours[j])))
            minMarkerDistancePixels = minimumPerimeter * min_marker_distance_rate
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


def _identify_candidates(gray, candidates, dictionary):
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
        valid, corners, cand_id = _identify_one_candidate(dictionary, gray, candidates[i])
        if valid:
            accepted.append(corners)
            ids.append(cand_id)
        else:
            rejected.append(corners)

    return accepted, ids, rejected


def _identify_one_candidate(dictionary, gray, corners):
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
    marker_border_bits = params[markerBorderBits]
    assert len(corners) is 4
    assert gray is not None
    assert marker_border_bits > 0

    # Get bits, and ensure there are not too many erroneous bits
    candidate_bits = _extract_bits(gray, corners)
    max_errors_in_border = int(dictionary.markerSize * dictionary.markerSize * marker_border_bits)
    border_errors = _get_border_errors(candidate_bits, dictionary.markerSize, marker_border_bits)
    if border_errors > max_errors_in_border:
        return False, corners, -1

    # Take inner bits for marker identification with beautiful Python slicing (god damn!)
    inner_bits = candidate_bits[marker_border_bits:-marker_border_bits, marker_border_bits:-marker_border_bits]
    retval, cand_id, rotation = dictionary.identify(inner_bits, params[errorCorrectionRate])
    if retval is False:
        return False, corners, -1
    else:
        # Shift corner positions to correct rotation before returning
        return True, np.roll(corners, rotation, axis=0), cand_id
    pass


def _extract_bits(gray, corners):
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
    marker_size = FiducialMarker.get_marker_size()  # size of inner region of marker (area containing ID information)
    marker_border_bits = params[markerBorderBits]  # size of marker border
    cell_size = params[perspectiveRemovePixelPerCell]  # size of "cell", area consisting of one bit of info.
    cell_margin_rate = params[perspectiveRemoveIgnoredMarginPerCell]  # cell margin
    min_std_dev_otsu = params[minOtsuStdDev]  # min. std. dev. needed to run Otsu thresholding

    # Run assertions
    assert len(gray.shape) == 2
    assert len(corners) == 4
    assert marker_border_bits > 0 and cell_size > 0 and cell_margin_rate >= 0 and cell_margin_rate <= 1
    assert min_std_dev_otsu >= 0

    # Determine new dimensions of perspective-removed marker
    markerSizeWithBorders = marker_size + 2*marker_border_bits
    cellMarginPixels = int(cell_margin_rate * cell_size)
    resultImgSize = int(markerSizeWithBorders * cell_size)
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
    inner_region = result_img[int(cell_size/2):int(-cell_size/2), int(cell_size/2):int(-cell_size/2)]

    # Check if standard deviation enough to apply Otsu thresholding
    # If not enough, probably means all bits are same color (black or white)
    mean, stddev = cv2.meanStdDev(inner_region)
    if stddev < min_std_dev_otsu:
        bits.fill(1) if mean > 127 else bits
        return bits

    # Because standard deviation is high enough, threshold using Otsu
    _, result_img = cv2.threshold(result_img, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    for y in range(markerSizeWithBorders):
        for x in range(markerSizeWithBorders):
            # Get each individual square of each cell, excluding the margin pixels
            yStart = y * cell_size + cellMarginPixels
            yEnd = yStart + cell_size - 2 * cellMarginPixels
            xStart = x * cell_size + cellMarginPixels
            xEnd = xStart + cell_size - 2 * cellMarginPixels
            square = result_img[yStart:yEnd, xStart:xEnd]
            if cv2.countNonZero(square) > (square.size / 2):
                bits[y][x] = 1
    return bits


def _get_border_errors(bits, marker_size, border_size):
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

def _filter_detected_markers(corners, ids):
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
            if i == j or ids[i] != ids[j]:
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
