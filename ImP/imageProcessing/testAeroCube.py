import unittest
import numpy as np
from .aerocubeMarker import AeroCubeMarker, AeroCubeFace, AeroCube


class TestAeroCube(unittest.TestCase):
    VALID_MARKERS = [
        AeroCubeMarker(
            0,
            AeroCubeFace.ZENITH,
            np.array([[[884.,  659.],
                       [812.,  657.],
                       [811.,  585.],
                       [885.,  586.]]])
        ),
        AeroCubeMarker(
            0,
            AeroCubeFace.FRONT,
            np.array([[[504.,  653.],
                       [433.,  653.],
                       [433.,  581.],
                       [505.,  582.]]])
        )
    ]
    INVALID_MARKERS_EMPTY = []
    INVALID_MARKERS_DIFFERENT_AEROCUBES = [
        AeroCubeMarker(
            0,
            AeroCubeFace.ZENITH,
            np.array([[[884.,  659.],
                       [812.,  657.],
                       [811.,  585.],
                       [885.,  586.]]])
        ),
        AeroCubeMarker(
            1,
            AeroCubeFace.FRONT,
            np.array([[[504.,  653.],
                       [433.,  653.],
                       [433.,  581.],
                       [505.,  582.]]])
        )
    ]

    def test_init(self):
        aerocube = AeroCube(self.VALID_MARKERS)
        self.assertIsNotNone(aerocube)
        self.assertEqual(aerocube.markers, self.VALID_MARKERS)

    def test_eq(self):
        aerocube_1 = AeroCube(self.VALID_MARKERS)
        aerocube_2 = AeroCube(self.VALID_MARKERS)
        self.assertEqual(aerocube_1, aerocube_2)

    def test_raise_if_markers_invalid(self):
        try:
            AeroCube.raise_if_markers_invalid(self.VALID_MARKERS)
        except Exception:
            self.fail()

    def test_err_raise_if_markers_invalid(self):
        self.assertRaises(AttributeError, AeroCube, self.INVALID_MARKERS_EMPTY)
        self.assertRaises(AttributeError, AeroCube, self.INVALID_MARKERS_DIFFERENT_AEROCUBES)


if __name__ == '__main__':
    unittest.main()
