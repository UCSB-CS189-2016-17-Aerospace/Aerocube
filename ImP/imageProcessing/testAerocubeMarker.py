from cv2 import aruco
import numpy
import unittest
from .aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCubeMarkerAttributeError
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker, IDOutOfDictionaryBoundError


class TestAeroCubeMarker(unittest.TestCase):
    VALID_CORNER_ARG = numpy.array([[[82.,  51.],
                                     [453., 51.],
                                     [454., 417.],
                                     [82.,  417.]]])
    VALID_CORNER_ARG_1 = numpy.array([[[82.,  52.],
                                     [453., 51.],
                                     [454., 417.],
                                     [82.,  417.]]])
    INVALID_CORNER_ARG = numpy.array([[[82.,  51.],
                                     [453., 51.],
                                     [454., 417.]]])

    def test_init(self):
        marker_obj = AeroCubeMarker(1, AeroCubeFace.LEFT, self.VALID_CORNER_ARG)
        self.assertEqual(marker_obj.aerocube_ID, 1)
        self.assertEqual(marker_obj.aerocube_face, AeroCubeFace.LEFT)
        self.assertTrue(numpy.array_equal(marker_obj.corners, self.VALID_CORNER_ARG))

    def test_invalid_init_parameters(self):
        self.assertRaises(AeroCubeMarkerAttributeError, AeroCubeMarker,
                          -1,  AeroCubeFace.LEFT,   self.VALID_CORNER_ARG)
        self.assertRaises(AeroCubeMarkerAttributeError, AeroCubeMarker,
                          0,                   1,   self.VALID_CORNER_ARG)
        self.assertRaises(AeroCubeMarkerAttributeError, AeroCubeMarker,
                          0,  AeroCubeFace.NADIR, self.INVALID_CORNER_ARG)

    def test_positive_eq(self):
        marker_1 = AeroCubeMarker(4, AeroCubeFace.FRONT, self.VALID_CORNER_ARG)
        marker_2 = AeroCubeMarker(4, AeroCubeFace.FRONT, self.VALID_CORNER_ARG)
        self.assertTrue(marker_1 == marker_2)

    def test_negative_eq(self):
        marker_1 = AeroCubeMarker(4, AeroCubeFace.FRONT, self.VALID_CORNER_ARG)
        marker_2 = AeroCubeMarker(3, AeroCubeFace.FRONT, self.VALID_CORNER_ARG)
        self.assertFalse(marker_1 == marker_2)
        marker_3 = AeroCubeMarker(4, AeroCubeFace.BACK, self.VALID_CORNER_ARG)
        self.assertFalse(marker_1 == marker_3)
        marker_4 = AeroCubeMarker(4, AeroCubeFace.FRONT, self.VALID_CORNER_ARG_1)
        self.assertFalse(marker_1 == marker_4)

    def test_valid_aerocube_ID(self):
        valid_IDs = range(0, 7)
        for ID in valid_IDs:
            self.assertTrue(
                AeroCubeMarker._valid_aerocube_ID(ID),
                "test_valid_aerocube_ID failed on {}".format(ID))
        invalid_IDs = [-1, 8, 9]
        for ID in invalid_IDs:
            self.assertFalse(
                AeroCubeMarker._valid_aerocube_ID(ID),
                "test_valid_aerocube_ID failed on {}".format(ID))

    def test_positive_get_aerocube_marker_IDs(self):
        marker_IDs = [6, 7, 8, 9, 10, 11]
        # due to Python name mangling for private method, must
        # prepend method call with class name:
        # http://stackoverflow.com/questions/17709040/calling-a-method-from-a-parent-class-in-python
        test_marker_IDs = AeroCubeMarker._get_aerocube_marker_IDs(1)
        self.assertTrue(numpy.array_equal(marker_IDs, test_marker_IDs),
                        "_get_aerocube_marker_IDs failed")

    def test_error_get_aerocube_marker_IDs(self):
        invalid_IDs = [-1, 8, 9]
        for ID in invalid_IDs:
            self.assertRaises(IDOutOfDictionaryBoundError, AeroCubeMarker._get_aerocube_marker_IDs, ID)
            # with self.assertRaises(IDOutOfDictionaryBoundError):
            #     AeroCubeMarker._get_aerocube_marker_IDs(ID)

    def test_positive_get_aerocube_marker_set(self):
        marker_IDs = AeroCubeMarker._get_aerocube_marker_IDs(1)
        marker_imgs = [FiducialMarker.draw_marker(ID) for ID in marker_IDs]
        self.assertTrue(numpy.array_equal(
                            marker_imgs,
                            AeroCubeMarker.get_aerocube_marker_set(1)),
                        "positive_get_aerocube_marker_set failed")

    def test_negative_get_aerocube_marker_set(self):
        """
        Verify that, given two different AeroCube IDs, the two sets of marker
        images do not share any images
        """
        marker_IDs = AeroCubeMarker._get_aerocube_marker_IDs(0)
        marker_imgs = [FiducialMarker.draw_marker(ID) for ID in marker_IDs]
        test_marker_imgs = AeroCubeMarker.get_aerocube_marker_set(1)
        for img in marker_imgs:
            for test_img in test_marker_imgs:
                self.assertFalse(numpy.array_equal(img, test_img),
                                 "negative_get_aerocube_marker_set failed")

    def test_identify_marker(self):
        aerocube_tuple = (1, AeroCubeFace.LEFT)
        test_tuple = AeroCubeMarker.identify_marker_ID(11)
        self.assertEqual(aerocube_tuple, test_tuple)

    def test_error_identify_marker(self):
        self.assertRaises(IDOutOfDictionaryBoundError, AeroCubeMarker.identify_marker_ID, -1)
        # with self.assertRaises(IDOutOfDictionaryBoundError):
        #     AeroCubeMarker.identify_marker_ID(-1)

if __name__ == '__main__':
    unittest.main()
