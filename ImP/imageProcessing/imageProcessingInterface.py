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
    :ivar _img_mat: holds the matrix represnetation of an image
    :ivar _dispatcher: dictionary mapping AeroCubeSignals to functions
    """
    _DICTIONARY = AeroCubeMarker.get_dictionary()

    def __init__(self, file_path):
        """
        Upon instantiation, use file_path to load the image for this ImageProcessor
        :param file_path: path to image to be processed
        """
        self._img_mat = self._load_image(file_path)
        self._dispatcher = {
            ImageEventSignal.IDENTIFY_AEROCUBES: self._identify_aerocubes
        }

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
        cal = CameraCalibration.PredefinedCalibration.ANDREW_IPHONE
        camera_matrix = cal.CAMERA_MATRIX
        dist_coeffs = cal.DIST_COEFFS
        # call aruco function
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners,
                                                       marker_length,
                                                       camera_matrix,
                                                       dist_coeffs)
        return (rvecs, tvecs)

    def _find_position(self):
        pass

    def scan_image(self, img_signal, op_params=None):
        """
        Describes the higher-level process of processing an image to
            (1) identify any AeroCubes, and
            (2) determine their relative attitude and position
        Takes the image signal param and passes it to the dispatcher, which calls a method depending on the signal given
        :param img_signal: a valid signal (from ImageEventSignal) that indicates the operation requested
        :param op_params: optional parameter where additional operation parameters can be given
        :return: results from the function called within the dispatcher
        """
        if img_signal not in ImageEventSignal:
            raise TypeError("Invalid signal for ImP")
        try:
            return self._dispatcher[img_signal](op_params)
        except KeyError:
            # img_signal is not defined for the dispatcher
            # TODO: how to handle?
            print("KeyError")
            pass
        except Exception as ex:
            # all other exceptions
            # TODO: how to handle?
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            pass

    def draw_fiducial_markers(self, corners, marker_IDs):
        """
        Returns an image matrix with the given corners and marker_IDs drawn onto the image
        :param corners: marker corners
        :param marker_IDs: fiducial marker IDs
        :return: img with marker boundaries drawn and markers IDed
        """
        return aruco.drawDetectedMarkers(self._img_mat, corners, marker_IDs)

    def draw_axis(self, cameraMatrix, distCoeffs, quaternion, tvec):
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
                              cameraMatrix, distCoeffs,
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
