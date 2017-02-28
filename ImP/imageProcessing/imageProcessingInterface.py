import itertools
import math
import cv2
from cv2 import aruco
import numpy as np
import pyquaternion
from jobs.aeroCubeSignal import ImageEventSignal
from .aerocubeMarker import AeroCubeMarker, AeroCube
from .aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCube
from .parallel import markerDetectPar as MarkerDetectPar
from .cameraCalibration import CameraCalibration
from .settings import ImageProcessingSettings


class ImageProcessor:
    """
    Reference on class docs: http://stackoverflow.com/questions/8649105/how-to-show-instance-attributes-in-sphinx-doc
    Instantiated with an image, provides the ability to process the image in various
    ways, most often by passing it AeroCubeSignal enum objects.
    :cvar _DICTIONARY: Aruco dictionary meant to be accessed only internally
    :ivar _img_mat: holds the matrix representation of an image
    :ivar _dispatcher: dictionary mapping AeroCubeSignals to functions
    """
    _DICTIONARY = AeroCubeMarker.get_dictionary()

    def __init__(self, file_path, cal=CameraCalibration.get_default_calibration()):
        """
        Upon instantiation, use file_path to load the image for this ImageProcessor
        :param file_path: path to image to be processed
        """
        self._img_mat = self._load_image(file_path)
        self._cal = cal

    @staticmethod
    def _load_image(file_path):
        """
        Method used to load an image given the file path (static since it
            does not rely on state).
        :param file_path: path used to find the image to be processed
        :return: the image specified as a matrix
        """
        image = cv2.imread(file_path)
        if image is None:
            raise OSError("cv2.imread returned None for path {}".format(file_path))
        return image

    # Aruco entry points

    def _find_fiducial_markers(self, gpu=False):
        """
        Identify fiducial markers in _img_mat
        Serves as an abstraction of the aruco method calls
        Note that the default format of the arrays returned by Aruco are a bit cumbersome, and are being translated
        into friendlier formats before being returned.
        :param gpu: optional param to attempt to use parallelized algorithm
        :return corners: an array of 3-D arrays
            each element is of the shape (N, 4, 2), where N is the number of detected markers
            If no markers found, corners == []
        :return marker_IDs: an array of integers corresponding to the corners.
            Note that the Aruco method returns a 1D numpy array of the form [id1, id2, ...],
            and has the shape (N,)
            If no markers found, marker_IDs == None
        """
        if gpu is True:
            corners, marker_IDs = MarkerDetectPar.detect_markers_parallel(self._img_mat, dictionary=self._DICTIONARY)
        else:
            corners, marker_IDs, _ = aruco.detectMarkers(self._img_mat, dictionary=self._DICTIONARY)
            corners, marker_IDs = self._simplify_fiducial_arrays(corners, marker_IDs)
        return corners, marker_IDs

    @staticmethod
    def _simplify_fiducial_arrays(corners, ids):
        """
        Translates the default arrays returned by Aruco's marker detection method into a simpler format.
        Corners: (N, 1, 4, 2) --> (N, 4, 2)
        IDs: (N, 1) --> (N,), same as 1-dimensional matrix/vector
        Note that when no markers are found, Aruco returns [] for corners and None for ids. In this case,
        return empty Numpy lists for both items.
        :param corners: corners formatted as (N, 1, 4, 2)
        :param ids: IDs formatted as (N, 1)
        :return: (corners, ids)
        """
        assert len(corners) is 0 or len(np.shape(corners)) is 4
        assert ids is None or len(np.shape(ids)) is 2
        if len(corners) == 0:
            return np.array(corners), np.array(list())
        else:
            return np.array(corners).squeeze(axis=1), np.squeeze(ids, axis=1)

    @staticmethod
    def _prepare_fiducial_arrays_for_aruco(corners, ids):
        """
        Translates simplified fiducial marker arrays to format usable by Aruco methods.
        Corners: (N, 4, 2) --> (N, 1, 4, 2)
        IDs: (N,) --> (N, 1)
        Note that when no markers are found, Aruco returns [] for corners and None for ids. Therefore, if
        params corners and ids are empty Numpy arrays, return ([], None)
        :param corners: corners formatted as (N, 4, 2)
        :param ids: IDs formatted as (N,)
        :return: (corners, ids)
        """
        assert len(corners) is 0 or len(np.shape(corners)) is 3
        assert len(ids) is 0 or len(np.shape(ids)) is 1
        if len(corners) == 0:
            return [], None
        else:
            return [np.array([c]) for c in corners], np.array([[i] for i in ids])

    def draw_fiducial_markers(self, corners, marker_IDs, img=None):
        """
        Returns an image matrix with the given corners and marker_IDs drawn onto the image
        :param corners: marker corners
        :param marker_IDs: fiducial marker IDs
        :param img:
        :return: img with marker boundaries drawn and markers IDed
        """
        img = np.copy(self._img_mat) if img is None else img
        aruco_corners, aruco_ids = self._prepare_fiducial_arrays_for_aruco(corners, marker_IDs)
        return aruco.drawDetectedMarkers(img, aruco_corners, aruco_ids)

    def draw_axis(self, quaternion, tvec, img=None):
        """
        Wrapper method that calls Aruco's draw axis method on a given marker.
        Can be used to visually verify the accuracy of pose.
        :param quaternion: pose represented as quaternion
        :param tvec: translation vector, returned by Aruco's estimatePoseSingleMarker
        :param img:
        :return: img held by this ImageProcessor with the drawn axis
        """
        img = np.copy(self._img_mat) if img is None else img
        return aruco.drawAxis(img,
                              self._cal.CAMERA_MATRIX,
                              self._cal.DIST_COEFFS,
                              self.quaternion_to_rodrigues(quaternion),
                              tvec,
                              ImageProcessingSettings.get_marker_length())

    def draw_aerocube_markers(self):
        markers = self._find_aerocube_markers()
        img_w_markers = self.draw_fiducial_markers([m.corners for m in markers], [m.aerocube_ID*AeroCube.NUM_SIDES+m.aerocube_face.value for m in markers])
        for m in markers:
            img_w_markers = self.draw_axis(m.quaternion, m.tvec, img=img_w_markers)
        return img_w_markers

    def draw_aerocubes(self):
        img_w_markers = self.draw_aerocube_markers()
        cubes = self._identify_aerocubes()
        for c in cubes:
            img_w_markers = self.draw_axis(c.quaternion, c.tvec, img=img_w_markers)
        return img_w_markers

    def _find_pose(self, corners):
        """
        Find the pose of identified markers.
        References:
            * http://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html
            * solvePnP: http://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
            * Perspective-n-Point: https://en.wikipedia.org/wiki/Perspective-n-Point
        :return rvecs: rotation vectors
        :return tvecs: translation vectors
        """
        # Get marker length
        marker_length = AeroCubeMarker.MARKER_LENGTH
        # Get camera calibration
        camera_matrix = self._cal.CAMERA_MATRIX
        dist_coeffs = self._cal.DIST_COEFFS
        # call aruco function
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners,
                                                       marker_length,
                                                       camera_matrix,
                                                       dist_coeffs)
        return rvecs, tvecs

    # AeroCube identification functions

    def _find_aerocube_markers(self, gpu=False):
        """
        Calls a private function to find all fiducial markers, then constructs
        AeroCubeMarker objects from those results. If there are no markers found,
        return an empty array.
        :return: array of AeroCubeMarker objects; empty if none found
        """
        corners, ids = self._find_fiducial_markers(gpu=gpu)
        if len(ids) is 0:
            return []
        else:
            rvecs, tvecs = self._find_pose(corners)
            quaternions = [self.rodrigues_to_quaternion(r) for r in rvecs]
            return [AeroCubeMarker(corners, id, q, tvec) for corners, id, q, tvec in zip(corners, ids, quaternions, tvecs)]

    def _identify_aerocubes(self):
        """
        Internal function called when ImP receives a ImageEventSignal.IDENTIFY_AEROCUBES signal.
        :return: array of AeroCube objects; [] if no AeroCubes found
        """
        markers = self._find_aerocube_markers()
        aerocubes = list()
        for aerocube, aerocube_markers in itertools.groupby(markers, lambda m: m.aerocube_ID):
            aerocubes.append(AeroCube(list(aerocube_markers)))
        return aerocubes

    def identify_markers_for_storage(self):
        # corners, ids = self._find_fiducial_markers()
        # rvecs, tvecs = self._find_pose(corners)
        # quaternions = [self.rodrigues_to_quaternion(r) for r in rvecs]
        # aeroCubeMarkers= list()
        # print("IMFS:ids from scan {}".format(ids))
        # for i in range(len(ids)):
        #     aeroCubeMarkers.append(AeroCubeMarker(ids[i][0],corners[i],quaternions[i],rvecs[i],tvecs[i]))
        # aeroCubes={}
        #
        # for aeroMarker in aeroCubeMarkers:
        #     IdKey=aeroMarker.aerocube_ID
        #     print("IMFS:IdKey is {}".format(IdKey))
        #     print("aerocubes.keys() {}".format(aeroCubes.keys()))
        #     print("IMFS: adding AeroCube {}".format(AeroCube(aeroMarker)))
        #     if(IdKey in aeroCubes.keys()):
        #         print("IMFS: multipul markers for same Aerocube")
        #         aeroCubes.get(IdKey).add_marker(aeroMarker)
        #     else:
        #         print("IMFS: new IdKey found {}".format(IdKey))
        #         aeroCubes[IdKey]=AeroCube(aeroMarker)
        # print("IMFS:aeroCubes is {}".format(aeroCubes))
        #
        # ids=list()
        # q_list = list()
        # for cube in aeroCubes.values():
        #     q_list.append({k: v for k, v in zip(['w', 'x', 'y', 'z'], cube.quaternion.elements)})
        #     ids.append(cube.ID)
        # return corners, ids, q_list
        return [marker.to_jsonifiable_dict() for marker in self._find_aerocube_markers()]

    # Pose and distance functions

    def _find_distance(self, corners):
        """
        Find the distance of an array of markers (represented by their corners).
        References:
        * http://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image
        * http://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
        :param corners: array of markers, each represented by their four corner points
        :param cal: calibration information of the camera used for the image
        :return: distance in meters
        """
        cal = self._cal
        # Find m (pixels per unit of measurement)
        m = (cal.CAMERA_MATRIX[0][0]/cal.FOCAL_LENGTH + cal.CAMERA_MATRIX[1][1]/cal.FOCAL_LENGTH)/2
        # Scale m for current resolution (if necessary), taking y information from original image and current
        m_for_res = self._img_mat.shape[0] * (m / cal.IMG_RES[0])
        # Initialize variables for loop
        dist_results = list()
        marker_size = ImageProcessingSettings.get_marker_length()
        for marker in corners:
            # TODO: can use diagonals instead
            pixel_length1 = math.sqrt(math.pow(marker[0][0] - marker[1][0], 2) + math.pow(marker[0][1] - marker[1][1], 2))
            pixel_length2 = math.sqrt(math.pow(marker[2][0] - marker[3][0], 2) + math.pow(marker[2][1] - marker[3][1], 2))
            pixlength = (pixel_length1+pixel_length2)/2
            dist = marker_size * cal.FOCAL_LENGTH / (pixlength/m_for_res)
            dist_results.append(dist)
        return dist_results

    @staticmethod
    def _find_distances_from_tvec(tvecs):
        """
        Finds the Euclidean distance of each translation vector in tvecs.
        :param tvecs:
        :return: Numpy array of floats representing Euclidean distance
        """
        return np.array([AeroCubeMarker.distance_from_tvec(tvec) for tvec in tvecs])

    @staticmethod
    def rodrigues_to_quaternion(rodrigues):
        """
        Converts an OpenCV rvec object (written in compact Rodrigues notation) into a quaternion.
        http://stackoverflow.com/questions/12933284/rodrigues-into-eulerangles-and-vice-versa
        :param rodrigues: rotation in compact Rodrigues notation (returned by cv2.Rodrigues) as 1x3 array
        :return: rotation represented as quaternion
        """
        return pyquaternion.Quaternion(matrix=cv2.Rodrigues(rodrigues)[0])

    @staticmethod
    def quaternion_to_rodrigues(quaternion):
        """
        Converts quaternion to rvec object (written in compact Rodrigues notation)
        :param quaternion: rotation represented as quaternion
        :return: rotation represented as rvec in compact Rodrigues notation
        """
        return cv2.Rodrigues(quaternion.rotation_matrix)[0]
