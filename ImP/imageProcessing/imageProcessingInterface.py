from aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCube
import cv2
from cv2 import aruco
import os


class ImageProcessor:
    _image_mat = None
    _DICTIONARY = AeroCubeMarker.get_dictionary()

    def __init__(self, file_path):
        self._image_mat = self._load_image(file_path)

    def _load_image(self, file_path):
        """
        :param file_path: Absolute path, from init argument,
        to load the image as a matrix into a variable
        :return:
        """
        image = cv2.imread(file_path)
        if image is None:
            raise OSError("cv2.imread returned None for path {}".format(file_path))
        return image

    def _find_fiducial_markers(self):
        """
        Identify fiducial markers in _image_mat
        Serves as an abstraction of the aruco method calls
        """
        (corners, marker_IDs, _) = aruco.detectMarkers(self._image_mat, dictionary=self._DICTIONARY)
        return (corners, marker_IDs)

    def _find_aerocube_markers(self):
        """
        Calls a private function to find all fiducial markers, then constructs
        AeroCubeMarker objects from those results
        """
        corners, marker_IDs = self._find_fiducial_markers()
        aerocube_IDs, aerocube_faces = zip(*[AeroCubeMarker.identify_marker_ID(ID) for ID in marker_IDs])
        aerocube_markers = list()
        for ID, face, marker_corners in zip(aerocube_IDs, aerocube_faces, corners):
            # because ID is in the form of [id_int], get the element
            aerocube_markers.append(AeroCubeMarker(ID[0], face, marker_corners))
        return aerocube_markers

    # TODO: given an array of AeroCubeMarker objects, return an array of
    # AeroCube objects with their respective AeroCubeMarker objects
    def _identify_aerocubes(self, aerocube_markers):
        """
        """
        pass

    def _find_attitude(self):
        pass

    def _find_position(self):
        pass

    def scan_image(self, img_signal):
        """
        Describes the higher-level process of processing an image to
        (1) identify any AeroCubes, and (2) determine their relative attitude and position
        :return:
        """
        """
        1. {
        imp = ImageProcessor(img_path)
        imp.scan_image(img_signal)
        } OR
        2. {
        scan_image(img_path, img_signal)
        }


        # assume that image is loaded from _load_image from __init__
        (corner_pts, marker_IDs, _) = imp._find_fiducial_markers(...)
        # identify aerocubes
        _find_aerocube_markers(...)
        _identify_aerocubes(...)
        # ask each aerocube to identify pose
        ...
        # return data (e.g., aerocube objects)
        ...
        """
        pass
