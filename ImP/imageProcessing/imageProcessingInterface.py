import itertools
import cv2
from cv2 import aruco
from jobs.aeroCubeSignal import ImageEventSignal
from .aerocubeMarker import AeroCubeMarker, AeroCube
import pyquaternion
import math
from .aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCube
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

    def __init__(self, file_path, cal=ImageProcessingSettings.get_default_calibration()):
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

    def _find_fiducial_markers(self):
        """
        Identify fiducial markers in _img_mat
        Serves as an abstraction of the aruco method calls
        :return corners: an array of 3-D arrays
            each element is of the form [[[ 884.,  659.],
                                          [ 812.,  657.],
                                          [ 811.,  585.],
                                          [ 885.,  586.]]]
            If no markers found, corners == []
        :return marker_IDs: an array of integers corresponding to the corners.
            Note that the Aruco method returns a 1D numpy array of the form [[id1], [id2], ...],
            and that elements must therefore be accessed as arr[idx][0], NOT arr[idx]
            If no markers found, marker_IDs == None
        """
        (corners, marker_IDs, _) = aruco.detectMarkers(self._img_mat, dictionary=self._DICTIONARY)
        return (corners, marker_IDs)

    def _find_aerocube_markers(self):
        """
        Calls a private function to find all fiducial markers, then constructs
        AeroCubeMarker objects from those results. If there are no markers found,
        return an empty array.
        :return: array of AeroCubeMarker objects; empty if none found
        """
        marker_corners, marker_IDs = self._find_fiducial_markers()
        if marker_IDs is None:
            return []
        else:
            aerocube_IDs, aerocube_faces = zip(*[AeroCubeMarker.identify_marker_ID(ID) for ID in marker_IDs])
            aerocube_markers = list()
            for ID, face, corners in zip(aerocube_IDs, aerocube_faces, marker_corners):
                # because ID is in the form of [id_int], get the element
                aerocube_markers.append(AeroCubeMarker(ID[0], face, corners))
            return aerocube_markers

    def _identify_aerocubes(self, *args, **kwargs):
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
        corners, ids = self._find_fiducial_markers()
        rvecs, tvecs = self._find_pose()
        quaternions = [self.rodrigues_to_quaternion(r) for r in rvecs]
        q_list = list()
        for q in quaternions:
            q_list.append({k: v for k, v in zip(['w', 'x', 'y', 'z'], q.elements)})
        return corners, ids, q_list

    # TODO: needs tests and perhaps better-defined behavior
    def _find_pose(self):
        """
        Find the pose of identified markers.
        References:
            * http://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html
            * solvePnP: http://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
            * Perspective-n-Point: https://en.wikipedia.org/wiki/Perspective-n-Point
        :return rvecs: rotation vectors
        :return tvecs: translation vectors
        """
        # get corners and marker length (from settings)
        corners, _ = self._find_fiducial_markers()
        marker_length = AeroCubeMarker.MARKER_LENGTH
        # get camera calibration
        camera_matrix = self._cal.CAMERA_MATRIX
        dist_coeffs = self._cal.DIST_COEFFS
        # call aruco function
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners,
                                                       marker_length,
                                                       camera_matrix,
                                                       dist_coeffs)
        return rvecs, tvecs

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

    def _find_position(self):
        pass

    def _identify_aerocubes_temp(self, *args, **kwargs):
        corners, ids = self._find_pose()

    def draw_fiducial_markers(self, corners, marker_IDs):
        """
        Returns an image matrix with the given corners and marker_IDs drawn onto the image
        :param corners: marker corners
        :param marker_IDs: fiducial marker IDs
        :return: img with marker boundaries drawn and markers IDed
        """
        return aruco.drawDetectedMarkers(self._img_mat, corners, marker_IDs)

    def draw_axis(self, quaternion, tvec):
        """
        Wrapper method that calls Aruco's draw axis method on a given marker.
        Can be used to visually verify the accuracy of pose.
        :param cameraMatrix: camera calibration matrix
        :param distCoeffs: camera distortion coefficients
        :param quaternion: pose represented as quaternion
        :param tvec: translation vector, returned by Aruco's estimatePoseSingleMarker
        :return: img held by this ImageProcessor with the drawn axis
        """
        return aruco.drawAxis(self._img_mat,
                              self._cal.CAMERA_MATRIX,
                              self._cal.DIST_COEFFS,
                              self.quaternion_to_rodrigues(quaternion),
                              tvec,
                              ImageProcessingSettings.get_marker_length())

    @staticmethod
    def rodrigues_to_quaternion(rodrigues):
        """
        Converts an OpenCV rvec object (written in compact Rodrigues notation) into a quaternion.
        http://stackoverflow.com/questions/12933284/rodrigues-into-eulerangles-and-vice-versa
        :param rodrigues: rotation in compact Rodrigues notation (returned by cv2.Rodrigues) as 1x3 array
        :return: rotation represented as quaternion
        """
        # theta = math.sqrt(rodrigues[0]**2 + rodrigues[1]**2 + rodrigues[2]**2)
        # quat = pyquaternion.Quaternion(scalar=theta, vector=[r/theta for r in rodrigues])
        quat = pyquaternion.Quaternion(matrix=cv2.Rodrigues(rodrigues)[0])
        return quat

    @staticmethod
    def quaternion_to_rodrigues(quaternion):
        """
        Converts quaternion to rvec object (written in compact Rodrigues notation)
        :param quaternion: rotation represented as quaternion
        :return: rotation represented as rvec in compact Rodrigues notation
        """
        return cv2.Rodrigues(quaternion.rotation_matrix)[0]
