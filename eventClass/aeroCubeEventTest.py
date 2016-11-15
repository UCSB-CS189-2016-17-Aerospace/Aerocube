from aeroCubeSignal import AeroCubeSignal
import unittest


class SignalTest(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

if __name__ == '__main__':
    unittest.main()
