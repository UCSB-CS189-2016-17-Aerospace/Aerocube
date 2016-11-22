from aeroCubeEvent import AeroCubeEvent, ImageEvent, ResultEvent
from aeroCubeSignal import AeroCubeSignal
import unittest


class TestAeroCubeEvent(unittest.TestCase):
    def test_image_event_init(self):
        event = ImageEvent(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_image_event_failed_init(self):
        self.assertRaises(AttributeError,
                          ImageEvent,
                          AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)


class TestAeroCubeSignal(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

if __name__ == '__main__':
    unittest.main()
