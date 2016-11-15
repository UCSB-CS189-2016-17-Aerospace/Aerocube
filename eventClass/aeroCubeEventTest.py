from signal import Signal
import unittest


class SignalTest(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(Signal.ImageEventSignal.IDENTIFY_AEROCUBES)

if __name__ == '__main__':
    unittest.main()
