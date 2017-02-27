from pyquaternion import Quaternion
from enum import Enum
import json
import numpy as np
from ImP.imageProcessing.settings import ImageProcessingSettings
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker, IDOutOfDictionaryBoundError


class AeroCubeMarker(FiducialMarker):
    MARKER_LENGTH = ImageProcessingSettings.get_marker_length()

    def __init__(self, corners, fiducial_marker_id, quaternion, tvec):
        # Aerocube ID and Face
        self._aerocube_ID, self._aerocube_face = self.identify_marker_ID(fiducial_marker_id)
        # Marker corners
        if corners.shape != (4, 2):
            raise AeroCubeMarkerAttributeError("Invalid corner matrix shape")
        else:
            self._corners = corners
        # Marker pose
        if not isinstance(quaternion, Quaternion):
            raise AeroCubeMarkerAttributeError("Invalid quaternion")
        else:
            self._quaternion = quaternion
        # Marker position (translation vector)
        self._tvec = tvec
        # Marker distance (derived attribute)
        self._distance = np.linalg.norm(tvec)

    def __eq__(self, other):
        if type(self) is type(other):
            return (self.aerocube_ID == other.aerocube_ID and
                    self.aerocube_face == other.aerocube_face and
                    np.array_equal(self.corners, other.corners))
        else:
            return False

    @property
    def aerocube_ID(self):
        return self._aerocube_ID

    @property
    def aerocube_face(self):
        return self._aerocube_face

    @property
    def corners(self):
        return self._corners

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def distance(self):
        return self._distance

    @staticmethod
    def _valid_aerocube_ID(aerocube_id):
        return (
            aerocube_id >= 0 and
            aerocube_id * AeroCube.NUM_SIDES + AeroCube.NUM_SIDES <= AeroCubeMarker.get_dictionary_size()
        )

    def to_json(self):
        json_dict = {
            "AEROCUBE_ID": self.aerocube_ID,
            "AEROCUBE_FACE": self.aerocube_face,
            "CORNERS": self.corners,
            "QUATERNION": {k: v for k, v in zip(['w', 'x', 'y', 'z'], self.quaternion.elements)},
            "DISTANCE": self.distance
        }
        return json.dumps(json_dict)

    @staticmethod
    def _get_aerocube_marker_IDs(aerocube_ID):
        """
        Get the list of marker IDs for a given AeroCube and it's ID
        Marker IDs are within the range [aerocube_ID*6, aerocube_ID*6 + 6],
        where aerocube IDs and marker IDs are 0 indexed
        :param aerocube_ID: ID of the AeroCube
        :return: array of marker IDs that can be used to attain marker images
        """
        if not AeroCubeMarker._valid_aerocube_ID(aerocube_ID):
            raise IDOutOfDictionaryBoundError('Invalid AeroCube ID(s)')
        base_marker_ID = aerocube_ID * AeroCube.NUM_SIDES
        end_marker_ID = base_marker_ID + AeroCube.NUM_SIDES
        return list(range(base_marker_ID, end_marker_ID))

    @staticmethod
    def get_aerocube_marker_set(aerocube_ID):
        marker_IDs = AeroCubeMarker._get_aerocube_marker_IDs(aerocube_ID)
        return [AeroCubeMarker.draw_marker(ID) for ID in marker_IDs]

    @staticmethod
    def identify_marker_ID(marker_ID):
        if marker_ID >= AeroCubeMarker.get_dictionary_size() or marker_ID < 0:
            raise IDOutOfDictionaryBoundError('Invalid Marker ID')
        aerocube_ID = marker_ID // AeroCube.NUM_SIDES
        aerocube_face = AeroCubeFace(marker_ID % AeroCube.NUM_SIDES)
        return aerocube_ID, aerocube_face


class AeroCubeFace(Enum):
    # Zenith is defined as the side facing away from the Earth
    # Nadir is defined as the side facing towards the Earth
    ZENITH = 0
    NADIR = 1
    FRONT = 2
    RIGHT = 3
    BACK = 4
    LEFT = 5

    def __init__(self, face_id):
        quaternions = {
            0: [.7071, .7071, 0, 0],
            1: [.7071, -.7071, 0, 0],
            2: [0, 0, 0, 0],
            3: [.7071, 0, -.7071, 0],  # these might be wrong
            4: [0, 0, 1, 0],
            5: [.7071, 0, .7071, 0] # these might be wrong
        }
        self.quaternion = Quaternion(quaternions[face_id])


class AeroCube:
    NUM_SIDES = 6

    # Give _ERR_MESSAGES keys unique, but otherwise arbitrary, values
    _MARKERS_EMPTY, _MARKERS_HAVE_MANY_AEROCUBES, _DUPLICATE_MARKERS = range(3)

    _ERR_MESSAGES = {
        _MARKERS_EMPTY:               "Markers for an AeroCube cannot be empty",
        _MARKERS_HAVE_MANY_AEROCUBES: "AeroCube Markers do not belong to same AeroCube (IDs are {})",
        _DUPLICATE_MARKERS:           "Duplicate AeroCube Marker used (Fiducial ID: {})"
    }

    def __init__(self, markers):
        # Check if arguments are valid
        self.raise_if_markers_invalid(markers)
        # Set instance variables
        print("making Aerocube")
        self._markers = markers
        self._ID = markers[0].aerocube_ID
        self._tvec = None
        self._quaternion = self.reduce_quaternions(markers)

    def __eq__(self, other):
        """
        Checks if two AeroCube objects are equivalent based on
            1. ID
            2. Identified markers
            3. Rotational vector(s)
            4. Translational vector(s)
        :return: boolean indicating equivalence of self and other
        """
        return self.ID == other.ID and \
            np.array_equal(self.markers, other.markers) and \
            np.array_equal(self.rvec, other.rvec) and \
            np.array_equal(self.tvec, other.tvec)

    @property
    def markers(self):
        return self._markers

    @property
    def ID(self):
        return self._ID

    @property
    def tvec(self):
        return self._tvec

    @property
    def quaternion(self):
        return self._quaternion

    def to_json(self):
        json_dict = {
            "MARKERS": [m.to_json() for m in self.markers],
            "ID": self.ID,
            "QUATERNION": self.quaternion
        }
        return json.dumps(json_dict)

    @staticmethod
    def reduce_quaternions(markers):
        candidate_quats = [marker.quaternion * marker.aerocube_face.quaternion for marker in markers]
        all_close = np.bitwise_and.reduce([np.allclose(candidate_quats[0].elements, q.elements) for q in candidate_quats], True)
        if all_close:
            return Quaternion(np.mean(candidate_quats, axis=0))
        else:
            raise AttributeError("Quaternions are not close enough for AeroCube!")

    @staticmethod
    def raise_if_markers_invalid(markers):
        """
        Tests if the given array of AeroCube Markers are a valid set to be input as
        constructor arguments for an AeroCube.
        If markers are invalid, raise an exception.
        Checks for the following condition:
            1. Markers is non-empty (an AeroCube object should not be created if there are no markers)
            2. Markers have identical AeroCube IDs
        :param markers: array of AeroCube Markers to be tested
        """
        if not markers:
            raise AttributeError(AeroCube._ERR_MESSAGES[AeroCube._MARKERS_EMPTY])
        if not all(marker.aerocube_ID == markers[0].aerocube_ID for marker in markers):
            aerocube_IDs = set([marker.aerocube_ID for marker in markers])
            raise AttributeError(AeroCube._ERR_MESSAGES[AeroCube._MARKERS_HAVE_MANY_AEROCUBES].format(aerocube_IDs))


class AeroCubeMarkerAttributeError(Exception):
    """
    Raised when an attribute of AeroCubeMarker is incorrectly assigned
    """
