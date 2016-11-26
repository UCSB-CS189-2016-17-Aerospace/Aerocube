from enum import Enum
import numpy
# relative imports are still troublesome -- temporary fix
# http://stackoverflow.com/questions/72852/how-to-do-relative-imports-in-python
import sys
sys.path.insert(1, '/home/ubuntu/GitHub/ImP')
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker


class AeroCubeMarker(FiducialMarker):
    _aerocube_ID = None
    _aerocube_face = None
    _corners = None
    _rvec = None  # rotation vector
    _tvec = None  # translation vector

    def __init__(self, aerocube_ID, aerocube_face, corners):
        self.aerocube_ID = aerocube_ID
        self.aerocube_face = aerocube_face
        self.corners = corners

    def __eq__(self, other):
        if type(self) is type(other):
            return (self.aerocube_ID == other.aerocube_ID and
                    self.aerocube_face == other.aerocube_face and
                    numpy.array_equal(self.corners, other.corners))
        else:
            return False

    @property
    def aerocube_ID(self):
        return self._aerocube_ID

    @aerocube_ID.setter
    def aerocube_ID(self, ID):
        if not self._valid_aerocube_ID(ID):
            raise AeroCubeMarkerAttributeError("Invalid AeroCube ID")
        self._aerocube_ID = ID

    @property
    def aerocube_face(self):
        return self._aerocube_face

    @aerocube_face.setter
    def aerocube_face(self, face):
        if not isinstance(face, AeroCubeFace):
            raise AeroCubeMarkerAttributeError("Invalid AeroCube face")
        self._aerocube_face = face

    @property
    def corners(self):
        return self._corners

    @corners.setter
    def corners(self, c):
        if c.shape != (1, 4, 2):
            raise AeroCubeMarkerAttributeError("Invalid corner matrix shape")
        self._corners = c

    @staticmethod
    def _valid_aerocube_ID(ID):
        return (
            ID >= 0 and
            ID*AeroCube.NUM_SIDES + AeroCube.NUM_SIDES <= AeroCubeMarker.get_dictionary_size()
        )

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
        return (aerocube_ID, aerocube_face)


class AeroCubeFace(Enum):
    # Zenith is defined as the side facing away from the Earth
    # Nadir is defined as the side facing towards the Earth
    ZENITH, NADIR, FRONT, RIGHT, BACK, LEFT = range(6)


class AeroCube():
    NUM_SIDES = 6
    _markers = None
    _rvec = None
    _tvec = None


class AeroCubeMarkerAttributeError(Exception):
    """
    Raised when an attribute of AeroCubeMarker is incorrectly assigned
    """
