import numpy
import unittest
from pyquaternion import Quaternion
import numpy as np
from ImP.imageProcessing.aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCubeMarkerAttributeError
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker, IDOutOfDictionaryBoundError


class TestAeroCubeMarker(unittest.TestCase):
    VALID_CORNER_ARG = numpy.array([[82.,  51.],
                                    [453., 51.],
                                    [454., 417.],
                                    [82.,  417.]])
    VALID_CORNER_ARG_1 = numpy.array([[82.,  52.],
                                      [453., 51.],
                                      [454., 417.],
                                      [82.,  417.]])
    INVALID_CORNER_ARG = numpy.array([[82.,  51.],
                                      [453., 51.],
                                      [454., 417.]])
    VALID_QUATERNION = Quaternion(1, 1, 1, 0)
    VALID_TVEC = np.array([1., 1., 1.])

    def test_init(self):
        marker_obj = AeroCubeMarker(self.VALID_CORNER_ARG, 7, self.VALID_QUATERNION, self.VALID_TVEC)
        self.assertEqual(marker_obj.aerocube_ID, 1)
        self.assertEqual(marker_obj.aerocube_face, AeroCubeFace.NADIR)
        np.testing.assert_array_equal(marker_obj.corners, self.VALID_CORNER_ARG)
        self.assertEqual(marker_obj.quaternion, self.VALID_QUATERNION)
        np.testing.assert_allclose(marker_obj.tvec, self.VALID_TVEC)
        self.assertIsNotNone(marker_obj.distance)

    def test_invalid_init_parameters(self):
        self.assertRaises(IDOutOfDictionaryBoundError, AeroCubeMarker,
                          self.VALID_CORNER_ARG, -1, self.VALID_QUATERNION, self.VALID_TVEC)
        self.assertRaises(AeroCubeMarkerAttributeError, AeroCubeMarker,
                          self.INVALID_CORNER_ARG, 1, self.VALID_QUATERNION, self.VALID_TVEC)
        self.assertRaises(AeroCubeMarkerAttributeError, AeroCubeMarker,
                          self.VALID_CORNER_ARG, 1, [1, 1, 1, 0], self.VALID_TVEC)

    def test_positive_eq(self):
        marker_1 = AeroCubeMarker(self.VALID_CORNER_ARG, 4, self.VALID_QUATERNION, self.VALID_TVEC)
        marker_2 = AeroCubeMarker(self.VALID_CORNER_ARG, 4, self.VALID_QUATERNION, self.VALID_TVEC)
        self.assertTrue(marker_1 == marker_2)
        marker_3 = AeroCubeMarker(self.VALID_CORNER_ARG, 3, self.VALID_QUATERNION, self.VALID_TVEC)
        marker_4 = AeroCubeMarker(self.VALID_CORNER_ARG_1, 4, self.VALID_QUATERNION, self.VALID_TVEC)
        self.assertFalse(marker_1 == marker_3)
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

    def test_as_jsonifiable_dict(self):
        self.fail()

    def test_distance_from_tvec(self):
        tvec = [ 0.08787901, -0.03494572,  0.8768408]
        np.testing.assert_allclose([AeroCubeMarker.distance_from_tvec(tvec)], [0.88192613806])

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
