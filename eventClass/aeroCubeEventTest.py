from aeroCubeEvent import AeroCubeEvent, ImageEvent, ResultEvent, SystemEvent
from aeroCubeSignal import AeroCubeSignal
import unittest


class AeroCubeEventTest(unittest.TestCase):
    def test_image_event_init(self):
        event = ImageEvent(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_image_event_init_failed(self):
        self.assertRaises(AttributeError,
                          ImageEvent,
                          AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)

    def test_result_event_init(self):
        event = ResultEvent(AeroCubeSignal.ResultEventSignal.IMP_OPERATION_OK)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_result_event_init_failed(self):
        self.assertRaises(AttributeError,
                          ResultEvent,
                          AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

    def test_system_event_init(self):
        event = SystemEvent(AeroCubeSignal.SystemEventSignal.POWERING_OFF)
        self.assertIsNotNone(event)
        self.assertIsNotNone(event.signal)
        self.assertIsNotNone(event.created_at)

    def test_system_event_init_failed(self):
        self.assertRaises(AttributeError,
                          SystemEvent,
                          AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)


class AeroCubeSignalTest(unittest.TestCase):

    def test_get_image_event_signal(self):
        self.assertIsNotNone(AeroCubeSignal.ImageEventSignal.IDENTIFY_AEROCUBES)

if __name__ == '__main__':
    unittest.main()
